import os
from io import BytesIO

import mlflow
import starlette
import torch
from PIL import Image
from ray import serve
from torchvision import transforms


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4})
class CellDetectionModel:
    def __init__(self):
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
        self.model = mlflow.pytorch.load(os.environ["MODEL_URI"])
        self.model.eval()
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(int(os.environ["IMG_LENGTH"])),
                transforms.ToTensor(),
            ]
        )

    async def __call__(self, request: starlette.requests.Request):
        image_payload_bytes = await request.body()
        image = Image.open(BytesIO(image_payload_bytes))

        input_tensor = self.preprocessor(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_class_id = output.argmax().item()
        predicted_class = self.ID_TO_CLASS_MAP[predicted_class_id]
        return {"prediction": predicted_class}


# 2: Bind and run the deployment locally
app = CellDetectionModel.bind()
serve.run(app)
