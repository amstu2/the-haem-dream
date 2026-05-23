import os
from io import BytesIO

import mlflow
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from ray import serve
from torchvision import transforms
from utils import mlflow_server_alive

app = FastAPI()

NUM_REPLICAS = int(os.environ["NUM_REPLICAS"])
NUM_CPUS = int(os.environ["NUM_CPUS"])
NUM_GPUS = int(os.environ["NUM_GPUS"])


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": NUM_CPUS, "num_gpus": NUM_GPUS},
)
@serve.ingress(app)
class CellDetectionModel:
    def __init__(self):
        MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
        if not mlflow_server_alive(MLFLOW_TRACKING_URI):
            raise ConnectionError(
                f"Can't connect to mlflow server ({MLFLOW_TRACKING_URI}) - ending..."
            )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.CLASS_TO_ID_MAP = {
            "Band Neutrophil": 0,
            "Basophil": 1,
            "Blast": 2,
            "Eosinophil": 3,
            "Erythroblast": 4,
            "Giant Platelet": 5,
            "Lymphocyte": 6,
            "Metamyelocyte": 7,
            "Monocyte": 8,
            "Myelocyte": 9,
            "Platelet Cluster": 10,
            "Reactive Lymphocyte": 11,
            "Segmented Neutrophil": 12,
        }
        self.ID_TO_CLASS_MAP = {v: k for k, v in self.CLASS_TO_ID_MAP.items()}

        MODEL_URI = os.environ["CHAMPION_MODEL_URI"]
        self.model = mlflow.pytorch.load_model(MODEL_URI)
        self.model.eval()
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(int(os.environ["IMG_LENGTH"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # https://pytorch.org/hub/pytorch_vision_densenet/
            ]
        )

    @app.post("/infer")
    def infer(self, file: UploadFile):
        data = file.file.read()
        image = Image.open(BytesIO(data))

        input_tensor = self.preprocessor(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_class_id = output.argmax().item()
        predicted_class = self.ID_TO_CLASS_MAP[predicted_class_id]
        return {"prediction": predicted_class}


app = CellDetectionModel.bind()
if __name__ == "__main__":
    serve.run(app)
