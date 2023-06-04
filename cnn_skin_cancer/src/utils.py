import os
import pickle
import time
from functools import wraps
from typing import Any, Callable, Tuple

import boto3
import numpy as np
import s3fs
from boto3.session import Session
from PIL import Image


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


class AWSSession:
    def __init__(self):
        self.__region_name = os.getenv("AWS_REGION")
        self.__aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.__aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.__aws_role_name = os.getenv("AWS_ROLE_NAME")

        self.__boto3_role_session = None
        self.__s3fs_session = None

    def _get_role_access(self, session: Session) -> Tuple[str, str, str]:
        sts = session.client("sts", region_name=self.__region_name)
        account_id = sts.get_caller_identity()["Account"]
        # https://www.learnaws.org/2022/09/30/aws-boto3-assume-role/
        response = sts.assume_role(
            RoleArn=f"arn:aws:iam::{account_id}:role/{self.__aws_role_name}",
            RoleSessionName=f"{self.__aws_role_name}-session",
        )
        return (
            response["Credentials"]["AccessKeyId"],
            response["Credentials"]["SecretAccessKey"],
            response["Credentials"]["SessionToken"],
        )

    def set_sessions(self):
        user_session = boto3.Session(
            region_name=self.__region_name,
            aws_access_key_id=self.__aws_access_key_id,
            aws_secret_access_key=self.__aws_secret_access_key,
        )
        (
            tmp_aws_access_key_id,
            tmp_aws_secret_access_key,
            tmp_aws_session_token,
        ) = self._get_role_access(user_session)
        self.__boto3_role_session = boto3.Session(
            region_name=self.__region_name,
            aws_access_key_id=tmp_aws_access_key_id,
            aws_secret_access_key=tmp_aws_secret_access_key,
            aws_session_token=tmp_aws_session_token,
        )
        self.__s3fs_session = s3fs.S3FileSystem(
            key=tmp_aws_access_key_id,
            secret=tmp_aws_secret_access_key,
            token=tmp_aws_session_token,
            anon=False,
        )

    # def get_sessions(self) -> Tuple[Session, Session]:
    #     return self.boto3_role_session, self.s3fs_session

    @timeit
    def upload_npy_to_s3(self, data: np.array, s3_bucket: str, file_key: str) -> None:
        with self.__s3fs_session.open(f"{s3_bucket}/{file_key}", "wb") as f:
            f.write(pickle.dumps(data))

    @timeit
    def download_npy_from_s3(self, s3_bucket: str, file_key: str) -> np.array:
        return np.load(
            self.__s3fs_session.open("{}/{}".format(s3_bucket, file_key)),
            allow_pickle=True,
        )

    def read_image_from_s3(self, s3_bucket: str, imname: str) -> np.array:
        s3client = self.__boto3_role_session.client("s3")
        keyname = imname.split(f"{s3_bucket}/", 1)[1]
        file_stream = s3client.get_object(Bucket=s3_bucket, Key=keyname)["Body"]
        np_image = Image.open(file_stream).convert("RGB")
        return np.asarray(np_image)

    def list_files_in_bucket(self, path: str) -> list:
        return self.__s3fs_session.ls(path)
