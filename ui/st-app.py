import requests
import streamlit as st
from PIL import Image

st.title("Cell Detection")

cell_image = st.file_uploader("Upload cell image", type="image/*")
if cell_image:
    image = Image.open(cell_image)
    st.image(image, caption="Uploaded Cell Image")

if st.button("Run Inference"):
    img_bytes = cell_image.getvalue()

    response = requests.post(
        # "http://cell-detector.ray-serve:8000",
        "http://localhost:8000",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
    )

    if response.status_code == 200:
        st.write("Results:", response.json())
    else:
        st.error(f"ERORR: {response.content}")
