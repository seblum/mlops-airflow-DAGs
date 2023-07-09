import os

from utils import AWSSession

bucket = os.getenv("AWS_BUCKET")
path_raw_data = f"s3://{bucket}/data/"

aws_session = AWSSession()
aws_session.set_sessions()

aws_session.list_files_in_bucket(path_raw_data)[-10:]
