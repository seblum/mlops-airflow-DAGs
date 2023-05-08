import streamlit as st
import requests
import pandas as pd
import io
import json
import os
from PIL import Image

st.header("MLOps Engineering Project")
st.subheader("Skin Cancer Detection")

# FastAPI endpoint
FASTAPI_SERVING_IP = os.getenv("FASTAPI_SERVING_IP")
FASTAPI_SERVING_PORT = os.getenv("FASTAPI_SERVING_PORT")
endpoint = f"http://{FASTAPI_SERVING_IP}:{FASTAPI_SERVING_PORT}/predict"


# check for pngs?
test_image = st.file_uploader("", type=["jpg"], accept_multiple_files=False)

if test_image:

    image = Image.open(test_image)
    image_file = io.BytesIO(test_image.getvalue())
    files = {"file": image_file}

    col1, col2 = st.columns(2)

    with col1:
        st.image(test_image, caption="", use_column_width="always")

    with col2:
        if st.button("Start Prediction"):
            with st.spinner("Prediction in Progress. Please Wait..."):
                output = requests.post(endpoint, files=files, timeout=8000)
            st.success("Success! Click the Download button below to retrieve prediction results (JSON format)")
            st.json(output.json())
            st.download_button(
                label="Download",
                data=json.dumps(output.json()),  # Download as JSON file object
                file_name="automl_prediction_results.json",
            )
