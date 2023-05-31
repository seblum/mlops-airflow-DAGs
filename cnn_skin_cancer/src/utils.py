import os
import pickle
import time
from functools import wraps
from typing import Any, Callable

import boto3
import numpy as np
import s3fs
from PIL import Image

# read_image = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ROLE_NAME = os.getenv("AWS_ROLE_NAME")


def _get_role_access(session):
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    response = sts.assume_role(
        RoleArn=f"arn:aws:iam::{account_id}:role/{AWS_ROLE_NAME}", RoleSessionName=f"{AWS_ROLE_NAME}-session"
    )
    return (
        response["Credentials"]["AccessKeyId"],
        response["Credentials"]["SecretAccessKey"],
        response["Credentials"]["SessionToken"],
    )


def _get_sessions():
    # https://www.learnaws.org/2022/09/30/aws-boto3-assume-role/
    user_session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    aws_access_key_id, aws_secret_access_key, aws_session_token = _get_role_access(user_session)
    boto3_role_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    s3fs_session = s3fs.S3FileSystem(
        key=aws_access_key_id, secret=aws_secret_access_key, token=aws_session_token, anon=False
    )
    return boto3_role_session, s3fs_session


boto3_session, s3fs_session = _get_sessions()


def timeit(func) -> Callable[..., Any]:
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"\nFunction {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@timeit
def upload_npy_to_s3(data: np.array, s3_bucket: str, file_key: str) -> None:
    with s3fs_session.open(f"{s3_bucket}/{file_key}", "wb") as f:
        f.write(pickle.dumps(data))


@timeit
def download_npy_from_s3(s3_bucket: str, file_key: str) -> np.array:
    return np.load(s3fs_session.open("{}/{}".format(s3_bucket, file_key)), allow_pickle=True)


def read_image_from_s3(s3_bucket: str, imname: str) -> np.array:
    s3client = boto3_session.client("s3")
    keyname = imname.split(f"{s3_bucket}/", 1)[1]
    file_stream = s3client.get_object(Bucket=s3_bucket, Key=keyname)["Body"]
    np_image = Image.open(file_stream).convert("RGB")
    return np.asarray(np_image)


def list_files_in_bucket(path: str) -> list:
    return s3fs_session.ls(path)
