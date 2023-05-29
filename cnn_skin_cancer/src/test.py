# test if all are the same
# testfile = download_npy_from_s3(aws_bucket=aws_bucket,file_key=f"{parent_path}/X_train.pkl")
# print((testfile==X_train).all())


# used for local testing
# np.save(f"{parent_path}/X_train.npy", X_train)
# np.save(f"{parent_path}/y_train.npy", y_train)
# np.save(f"{parent_path}/X_test.npy", X_test)
# np.save(f"{parent_path}/y_test.npy", y_test)


# used for local testing
# X_train = np.load(f"{path_X_train}")
# y_train = np.load(f"{path_y_train}")
# X_test = np.load(f"{path_X_test}")
# y_test = np.load(f"{path_y_test}")


# used for local testing
# dir_path = pathlib.Path(__file__).parent.absolute()
# print(dir_path)
# parent_path = dir_path.parent
# print(parent_path)

# https://stackoverflow.com/questions/48049557/how-to-write-npy-file-to-s3-directly
# read_image = lambda imname: np.asarray(Image.open(imname).convert("RGB"))


# docker build -f cnn_skin_cancer/Docker/data-preprocessing/Dockerfile -t preprocessed:v1 .
