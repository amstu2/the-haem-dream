import os
import random
from pathlib import Path

import mlflow
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from ray.data.datasource.partitioning import Partitioning
from ray.data.preprocessors import LabelEncoder
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torchvision import transforms
from torchvision.models import densenet121
from utils import (
    download_gcs_file,
    mlflow_server_alive,
    unzip_file,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TRAIN_LEARNING_RATE = float(os.getenv("TRAIN_LEARNING_RATE", 0.001))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
EPOCHS = int(os.getenv("EPOCHS", 10))
DATASET_BASE_PATH = os.getenv(
    "DATASET_BASE_PATH", "gs://the-haem-dream/cell-dataset/extracted/dataset/"
)

IMG_LENGTH = int(os.getenv("IMG_LENGTH", 368))
PROBABILITY_VERT_FLIP = float(os.getenv("PROBABILITY_VERT_FLIP", 0.5))
PROBABILITY_HORI_FLIP = float(os.getenv("PROBABILITY_HORI_FLIP", 0.5))
PROBABILITY_BRIGHTNESS = float(os.getenv("PROBABILITY_BRIGHTNESS", 0.2))
PROBABILITY_CONTRAST = float(os.getenv("PROBABILITY_CONTRAST", 0.2))
TRAIN_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", 2))
TRAIN_USE_GPU = os.getenv("TRAIN_USE_GPU", "False").lower() == "true"
SEED = int(os.getenv("TRAIN_SEED", 42))
TRUNCATE_DATASET = os.getenv("TRUNCATE_DATASET", "False").lower() == "true"

if not mlflow_server_alive(MLFLOW_TRACKING_URI):
    raise ConnectionError(
        f"Can't connect to mlflow server ({MLFLOW_TRACKING_URI}- ending..."
    )


download_gcs_file("the-haem-dream", "cell-dataset/pbc_dataset.zip", "./pbc_dataset.zip")
download_gcs_file("the-haem-dream", "cell-dataset/pbc_meta.csv", "./pbc_meta.csv")

unzip_file(Path("./pbc_dataset.zip"), Path("./"))


train_root = DATASET_BASE_PATH + "train"
test_root = DATASET_BASE_PATH + "test"
train_partitioning = Partitioning("dir", field_names=["class"], base_dir=train_root)
test_partitioning = Partitioning("dir", field_names=["class"], base_dir=test_root)
train_ds = ray.data.read_images(
    train_root, size=(IMG_LENGTH, IMG_LENGTH), partitioning=train_partitioning
)
test_ds = ray.data.read_images(
    test_root, size=(IMG_LENGTH, IMG_LENGTH), partitioning=test_partitioning
)

encoder = LabelEncoder(label_column="class")
train_ds = encoder.fit_transform(train_ds)
test_ds = encoder.transform(test_ds)
if TRUNCATE_DATASET:  # For debugging
    train_ds = train_ds.limit(100)
    test_ds = test_ds.limit(10)


def train_func(config):
    # TODO: Switch to GCS native storage
    if ray.train.get_context().get_world_rank() == 0:
        mlflow.set_tracking_uri(config["mflow_server_uri"])
        mlflow.set_experiment("cell-detection")
        mlflow.start_run()
        mlflow.log_params(config)

    train_data_shard = ray.train.get_dataset_shard("train")
    test_data_shard = ray.train.get_dataset_shard("test")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {'cuda' if torch.cuda.is_available() else 'cpu'}")

    transformations = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=config["prob_vert_flip"]),
            transforms.RandomHorizontalFlip(p=config["prob_hori_flip"]),
            transforms.ColorJitter(
                brightness=config["prob_brightness"], contrast=config["prob_contrast"]
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # https://pytorch.org/hub/pytorch_vision_densenet/
        ]
    )

    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=config["batch_size"],
        dtypes={"image": torch.uint8, "class": torch.int64},
        device=device,
        local_shuffle_buffer_size=config["batch_size"]
        * 4,  # Hopefully should fit on 16Gi RAM
    )

    test_dataloader = test_data_shard.iter_torch_batches(
        batch_size=config["batch_size"],
        dtypes={"image": torch.uint8, "class": torch.int64},
        device=device,
    )

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, config["num_classes"])
    model.to(device)

    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    checkpoint_dir = Path("./checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True)

    model.train()
    best_loss = 1e8
    for epoch in range(config["epochs"]):
        batch_loss = 0.0
        batch_count = 0
        for batch in train_dataloader:
            classes = batch["class"]
            images = batch["image"] / 255.0
            images = images.permute(0, 3, 1, 2)
            images = transformations(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, classes)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_count += 1

        checkpoint = None
        epoch_loss = batch_loss / batch_count
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if ray.train.get_context().get_world_rank() == 0:
                formatted_model = model.module if hasattr(model, "module") else model
                torch.save(formatted_model.state_dict(), f"{checkpoint_dir}/model.pth")
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
        if ray.train.get_context().get_world_rank() == 0:
            mlflow.log_metrics({"loss": epoch_loss}, step=epoch)
        ray.train.report(metrics={"loss": epoch_loss}, checkpoint=checkpoint)

    model.eval()
    tp = torch.zeros(config["num_classes"]).to(device)
    fp = torch.zeros(config["num_classes"]).to(device)
    fn = torch.zeros(config["num_classes"]).to(device)
    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for batch in test_dataloader:
            classes = batch["class"]
            images = batch["image"] / 255.0
            images = images.permute(0, 3, 1, 2)

            outputs = model(images).argmax(1)

            for c in range(config["num_classes"]):
                tp[c] += ((outputs == c) & (classes == c)).sum()
                fp[c] += ((outputs == c) & (classes != c)).sum()
                fn[c] += ((outputs != c) & (classes == c)).sum()

            correct += (outputs == classes).sum()
            total += classes.size(0)

    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(fn, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    accuracy = 0
    macro_f1 = 0
    if ray.train.get_context().get_world_rank() == 0:
        EPSILON = 1e-8
        per_class_f1 = 2 * tp / (2 * tp + fp + fn + EPSILON)
        macro_f1 = per_class_f1.mean().item()
        accuracy = (correct / total).item()
        mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1})
        formatted_model = model.module if hasattr(model, "module") else model
        mlflow.pytorch.log_model(formatted_model, "model")
        mlflow.end_run()
    ray.train.report({"accuracy": accuracy, "macro_f1": macro_f1})


trainer = TorchTrainer(
    train_func,
    datasets={"train": train_ds, "test": test_ds},
    scaling_config=ScalingConfig(num_workers=TRAIN_NUM_WORKERS, use_gpu=TRAIN_USE_GPU),
    run_config=RunConfig(storage_path="gs://the-haem-dream/cell-detector-run-config"),
    train_loop_config={
        "mflow_server_uri": MLFLOW_TRACKING_URI,
        "lr": TRAIN_LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_classes": 13,
        "epochs": EPOCHS,
        "seed": SEED,
        "prob_vert_flip": PROBABILITY_VERT_FLIP,
        "prob_hori_flip": PROBABILITY_HORI_FLIP,
        "prob_brightness": PROBABILITY_BRIGHTNESS,
        "prob_contrast": PROBABILITY_CONTRAST,
    },
)
result = trainer.fit()
