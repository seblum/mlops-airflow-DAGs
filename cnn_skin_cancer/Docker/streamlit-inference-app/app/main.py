import io
import json
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.header("MLOps Engineering Project")
st.subheader("Skin Cancer Detection")

# FastAPI endpoint
FASTAPI_SERVING_IP = os.getenv("FASTAPI_SERVING_IP")
FASTAPI_SERVING_PORT = os.getenv("FASTAPI_SERVING_PORT")
FASTAPI_ENDPOINT = f"http://{FASTAPI_SERVING_IP}:{FASTAPI_SERVING_PORT}/predict"

# check for pngs?
test_image = st.file_uploader("", type=["jpg"], accept_multiple_files=False)

if test_image:
    image = Image.open(test_image)
    image_file = io.BytesIO(test_image.getvalue())
    files = {"file": image_file}

    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image in the first column
        st.image(test_image, caption="", use_column_width="always")

    with col2:
        if st.button("Start Prediction"):
            with st.spinner("Prediction in Progress. Please Wait..."):
                # Send a POST request to FastAPI for prediction
                output = requests.post(FASTAPI_ENDPOINT, files=files, timeout=8000)
            st.success("Success! Click the Download button below to retrieve prediction results (JSON format)")
            # Display the prediction results in JSON format
            st.json(output.json())
            # Add a download button to download the prediction results as a JSON file
            st.download_button(
                label="Download",
                data=json.dumps(output.json()),
                file_name="cnn_skin_cancer_prediction_results.json",
            )
