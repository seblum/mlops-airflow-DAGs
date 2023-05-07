import requests
import numpy as np
from PIL import Image
FASTAPI_SERVING_IP = "localhost"
FASTAPI_SERVING_PORT = "80"

# https://github.com/tiangolo/fastapi/issues/2376

endpoint = f'http://{FASTAPI_SERVING_IP}:{FASTAPI_SERVING_PORT}/predict'

file_path = "/Users/sebastian.blum/Documents/Personal/mlops-airflow-DAGs/test/data/3.jpg"
# file = np.asarray(Image.open(file_path).convert("RGB"))
# image = np.array(file, dtype="uint8")
# data = image / 255
# files = {'media': data.tolist()}
#print(open(file_path, 'rb'))
files = {'file': open(file_path, 'rb')}


#print(files)
#print(files)
output = requests.post(endpoint, files=files, timeout=8000)
print(output)
