import os
import random
from pathlib import Path

import mlflow
import ray
import torch
import torch.nn as nn
from ray.data.datasource.partitioning import Partitioning
from ray.data.preprocessors import LabelEncoder
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torchvision import transforms
from torchvision.models import densenet121

from the_haem_dream.utils import download_gcs_file, mlflow_server_alive, unzip_file

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
TRAIN_LEARNING_RATE = float(os.environ["TRAIN_LEARNING_RATE"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
EPOCHS = int(os.environ["EPOCHS"])

IMG_LENGTH = int(os.getenv("IMG_LENGTH", 368))
PROBABILITY_VERT_FLIP = float(os.getenv("PROBABILITY_VERT_FLIP", 0.5))
PROBABILITY_HORI_FLIP = float(os.getenv("PROBABILITY_HORI_FLIP", 0.5))
PROBABILITY_BRIGHTNESS = float(os.getenv("PROBABILITY_BRIGHTNESS", 0.2))
PROBABILITY_CONTRAST = float(os.getenv("PROBABILITY_CONTRAST", 0.2))
SEED = int(os.getenv("TRAIN_SEED", 42))
TRUNCATE_DATASET = bool(os.getenv("TRUNCATE_DATASET", False))

if not mlflow_server_alive(MLFLOW_TRACKING_URI):
    raise ConnectionError(
        f"Can't connect to mlflow server ({MLFLOW_TRACKING_URI}- ending..."
    )

ray.init()

mlflow.set_experiment("cell-detection")
mlflow_server_uri = os.environ["MLFLOW_TRACKING_URI"]
if not mlflow_server_alive(mlflow_server_uri):
    print(f"Can't connect to mlflow server ({mlflow_server_uri}- ending...")
    sys.exit(1)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path("C:\\Users\\andre\\Downloads\\the-haem-dream-246d90e23cca.json")
)

download_gcs_file("the-haem-dream", "cell-dataset/pbc_dataset.zip", "./pbc_dataset.zip")
download_gcs_file("the-haem-dream", "cell-dataset/pbc_meta.csv", "./pbc_meta.csv")

unzip_file(Path("./pbc_dataset.zip"), Path("./"))

pbc_base_path = Path("./dataset/").resolve()
train_root = pbc_base_path / "train"
train_root = pbc_base_path / "train"
partitioning = Partitioning("dir", field_names=["class"], base_dir=train_root)
train_ds = ray.data.read_images(
    train_root, size=(IMG_LENGTH, IMG_LENGTH), partitioning=partitioning
)

encoder = LabelEncoder(label_column="class")
train_ds = encoder.fit_transform(train_ds)
if TRUNCATE_DATASET:
    train_ds = train_ds.limit(100)  # For debugging


def train_func(config):
    data_shard = ray.train.get_dataset_shard("train")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    transformations = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=config["prob_vert_flip"]),
            transforms.RandomHorizontalFlip(p=config["prob_hori_flip"]),
            transforms.ColorJitter(
                brightness=config["prob_brightness"], contrast=config["prob_contrast"]
            ),
        ]
    )

    dataloader = data_shard.iter_torch_batches(
        batch_size=config["batch_size"],
        dtypes={"image": torch.uint8, "class": torch.int64},
    )

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, config["num_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    checkpoint_dir = Path("./checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True)

    model.train()
    for epoch in range(config["epochs"]):
        for batch in dataloader:
            classes = batch["class"]
            images = batch["image"] / 255.0
            images = images.permute(0, 3, 1, 2)
            images = transformations(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, classes)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth")
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            ray.train.report(metrics={"loss": loss.item()}, checkpoint=checkpoint)

        torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth")
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        ray.train.report(metrics={"loss": loss.item()}, checkpoint=checkpoint)


with mlflow.start_run():
    run_config_path = str(Path("run_config/").absolute())
    trainer = TorchTrainer(
        train_func,
        datasets={"train": train_ds},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        run_config=RunConfig(storage_path=run_config_path),
        train_loop_config={
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

ray.shutdown()
