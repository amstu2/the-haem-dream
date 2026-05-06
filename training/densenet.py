import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121
import ray
from ray.train import ScalingConfig, RunConfig
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
import pathlib
import numpy as np
from ray.data.datasource.partitioning import Partitioning

pbc_base_path = (pathlib.Path(__file__) / "../../data/pbc_dataset/dataset/").resolve()
train_root = pbc_base_path / "train"

ray.init()

train_root = pbc_base_path / "train"
partitioning = Partitioning("dir", field_names=["class"], base_dir=train_root)
train_ds = ray.data.read_images(train_root, size=(368, 368), partitioning=partitioning)
train_ds.schema()


def train_func(config):
    data_shard = ray.train.get_dataset_shard("train")

    transformations = transforms.Compose[
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]

    data_shard.map_batches(transformations)

    model = densenet121(weights=None)
    model.classifier = nn.Linear(config["input_features"], config["num_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    model.train()
    for epoch in range(config["epochs"]):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        checkpoint_dir = "checkpoints"
    torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth")
    checkpoint = Checkpoint.from_directory(checkpoint_dir)
    train.report(metrics={"loss": loss}, checkpoint=checkpoint)


run_config_path = str(pathlib.Path("run_config/").absolute())
trainer = TorchTrainer(
    train_func,
    datasets={"train": train_ds.limit(50)},
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    run_config=RunConfig(storage_path=run_config_path),
    train_loop_config={
        "lr": 1e-3,
        "input_features": (368 * 368 * 3),
        "num_classes": 13,
        "epochs": 200,
    },
)
result = trainer.fit()
