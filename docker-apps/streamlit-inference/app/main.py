import streamlit as st
import requests
import pandas as pd
import io
import json
import os 

st.title('MLOps Engineering Project: Skin Cancer Detection')

# FastAPI endpoint
#endpoint = 'http://localhost:8000/predict' # FASTAPIENDPOINT
FASTAPI_SERVING_IP = os.getenv("FASTAPI_SERVING_IP")
FASTAPI_SERVING_PORT = os.getenv("FASTAPI_SERVING_PORT")
endpoint = f'http://{FASTAPI_SERVING_IP}:{FASTAPI_SERVING_PORT}/predict'

test_csv = st.file_uploader('', type=['csv','xlsx'], accept_multiple_files=False)

if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('View Sample of Test Set')
    st.write(test_df.head())

    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)  # write to BytesIO buffer
    test_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError
    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    if st.button('Start Prediction'):
        with st.spinner('Prediction in Progress. Please Wait...'):
            output = requests.post(endpoint, files=files, timeout=8000)
        st.success('Success! Click the Download button below to retrieve prediction results (JSON format)')
        st.download_button(
            label='Download',
            data=json.dumps(output.json()), # Download as JSON file object
            file_name='automl_prediction_results.json'
        )