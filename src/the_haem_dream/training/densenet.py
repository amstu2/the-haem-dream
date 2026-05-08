import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121
import ray
from ray.train import ScalingConfig, RunConfig, Checkpoint
from ray.train.torch import TorchTrainer
from pathlib import Path
from ray.data.datasource.partitioning import Partitioning
from ray.data.preprocessors import LabelEncoder
from the_haem_dream.utils import download_gcs_file, unzip_file

ray.init()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path("C:\\Users\\andre\\Downloads\\the-haem-dream-246d90e23cca.json"))

download_gcs_file("the-haem-dream", "cell-dataset/pbc_dataset.zip", "./pbc_dataset.zip")
download_gcs_file("the-haem-dream", "cell-dataset/pbc_meta.csv", "./pbc_meta.csv")

unzip_file(Path("./pbc_dataset.zip"), Path("./"))

pbc_base_path = Path("./dataset/").resolve()
train_root = pbc_base_path / "train"
train_root = pbc_base_path / "train"
partitioning = Partitioning("dir", field_names=["class"], base_dir=train_root)
train_ds = ray.data.read_images(train_root, size=(368, 368), partitioning=partitioning)

encoder = LabelEncoder(label_column="class")
train_ds = encoder.fit_transform(train_ds)

def train_func(config:
    data_shard = ray.train.get_dataset_shard("train")

    transformations = transforms.Compose(
        [
            #transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )

    dataloader = data_shard.iter_torch_batches(
        batch_size=config["batch_size"], dtypes={"image": torch.uint8, "class": torch.int64}
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


run_config_path = str(Path("run_config/").absolute())
trainer = TorchTrainer(
    train_func,
    datasets={"train": train_ds.limit(50)},
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    run_config=RunConfig(storage_path=run_config_path),
    train_loop_config={
        "lr": 1e-3,
        "batch_size": 16,
        "num_classes": 13,
        "epochs": 10,
    },
)
result = trainer.fit()

ray.shutdown()
