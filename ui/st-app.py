import io
import os

import requests
import streamlit as st
from PIL import Image

st.title("Cell Detection")

MODEL_SERVER_URL = os.environ["MODEL_SERVER_URL"]

image = None
cell_image = st.file_uploader("Upload cell image", type="image/*")
if cell_image:
    image = Image.open(cell_image)
    st.image(image, caption="Uploaded Cell Image")

if st.button("Run Inference"):
    buf = io.BytesIO()
    if image is not None:
        image.save(buf, format="jpg")

        img_bytes = buf.getvalue()

        response = requests.post(
            MODEL_SERVER_URL,  # "http://cell-detector.ray-serve:8000/infer"
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
        )

        if response.status_code == 200:
            st.write("Results:", response.json())
        else:
            st.error(f"ERORR: {response.content}")
