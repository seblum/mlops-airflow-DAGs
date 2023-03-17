import os
import numpy as np
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding

DATAPATH = '../data/'

folder_benign_train = f'{DATAPATH}train/benign'
folder_malignant_train = f'{DATAPATH}train/malignant'

folder_benign_test = f'{DATAPATH}test/benign'
folder_malignant_test = f'{DATAPATH}test/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

def load_and_convert_images(folder_path:str):
    ims = [read(os.path.join(folder_path, filename)) for filename in os.listdir(folder_path)]
    return np.array(ims, dtype='uint8')

def create_label(x_dataset:np.array):
    return np.zeros(x_dataset.shape[0])

def merge_data(set_one:np.array,set_two:np.array):
    return np.concatenate((set_one, set_two), axis = 0)

# Load in training pictures 
X_benign = load_and_convert_images(folder_benign_train)
X_malignant = load_and_convert_images(folder_malignant_train)
X_train = merge_data(X_benign, X_malignant)

# Load in testing pictures
X_benign_test = load_and_convert_images(folder_benign_test)
X_malignant_test = load_and_convert_images(folder_malignant_test)
X_test = merge_data(X_benign_test, X_malignant_test)

# Create labels
y_benign = create_label(X_benign)
y_malignant = create_label(X_malignant)
y_train = merge_data(y_benign, y_malignant)

y_benign_test = create_label(X_benign_test)
y_malignant_test = create_label(X_malignant_test)
y_test = merge_data(y_benign_test, y_malignant_test)

from sklearn.utils import shuffle
# Shuffle data
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

y_train = to_categorical(y_train, num_classes= 2)
y_test = to_categorical(y_test, num_classes= 2)

# With data augmentation to prevent overfitting 
X_train = X_train/255.
X_test = X_test/255.


# Shuffle data
# s = np.arange(X_train.shape[0])
# np.random.shuffle(s)
# X_train = X_train[s]
# y_train = y_train[s]

# s = np.arange(X_test.shape[0])
# np.random.shuffle(s)
# X_test = X_test[s]
# y_test = y_test[s]



# # Display first 15 images of moles, and how they are classified
# # This image can be logged and stored in mlflow
# w=40
# h=30
# fig=plt.figure(figsize=(12, 8))
# columns = 5
# rows = 3

# for i in range(1, columns*rows +1):
#     ax = fig.add_subplot(rows, columns, i)
#     if y_train[i] == 0:
#         ax.title.set_text('Benign')
#     else:
#         ax.title.set_text('Malignant')
#     plt.imshow(X_train[i], interpolation='nearest')
# plt.show()
# plt.savefig('temp.png', dpi=fig.dpi)


y_train = to_categorical(y_train, num_classes= 2)
y_test = to_categorical(y_test, num_classes= 2)


# With data augmentation to prevent overfitting 
X_train = X_train/255.
X_test = X_test/255.

