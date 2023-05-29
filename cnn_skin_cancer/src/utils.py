import pickle
import time
from functools import wraps
from typing import Any, Callable

import boto3
import numpy as np
import pandas as pd
import s3fs
from PIL import Image
from s3fs.core import S3FileSystem

s3 = S3FileSystem()
session = boto3.Session()

# read_image = lambda imname: np.asarray(Image.open(imname).convert("RGB"))


def timeit(func) -> Callable[..., Any]:
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@timeit
def upload_npy_to_s3(data: np.array, s3_bucket: str, file_key: str) -> None:
    with s3.open(f"{s3_bucket}/{file_key}", "wb") as f:
        f.write(pickle.dumps(data))


@timeit
def download_npy_from_s3(s3_bucket: str, file_key: str) -> np.array:
    return np.load(s3.open("{}/{}".format(s3_bucket, file_key)), allow_pickle=True)


def read_image_from_s3(s3_bucket: str, imname: str) -> np.array:
    s3client = session.client("s3")
    keyname = imname.split(f"{s3_bucket}/", 1)[1]
    file_stream = s3client.get_object(Bucket=s3_bucket, Key=keyname)["Body"]
    np_image = Image.open(file_stream).convert("RGB")
    return np.asarray(np_image)


def list_files_in_bucket(path: str) -> list:
    # TODO: check whether profile is needed
    s3 = s3fs.S3FileSystem(profile="seblum", anon=False)
    return s3.ls(path)
