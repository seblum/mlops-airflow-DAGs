from keras.applications import resnet50
from keras.models import Model
from keras.optimizers import Adam

# from tensorflow.keras.applications import resnet50


class ResNet50:
    def __init__(self, model_params: dict):
        self.model = resnet50.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=model_params.get("input_shape"),
            pooling=model_params.get("pooling"),
            classes=model_params.get("num_classes"),
        )

    def call(self, model_params: dict) -> Model:
        return self.model.compile(
            optimizer=Adam(model_params.get("learning_rate")),
            loss=model_params.get("loss"),
            metrics=model_params.get("metrics"),
        )


# https://towardsdatascience.com/tensorflow-class-inheritance-beautiful-code-59d2eb7cdfce
# https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e


# class LeNet5(tf.keras.Model):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         # creating layers in initializer
#         self.conv1 = Conv2D(filters=6, kernel_size=(5, 5), padding="same", activation="relu")
#         self.max_pool2x2 = MaxPool2D(pool_size=(2, 2))
#         self.conv2 = Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation="relu")
#         self.conv3 = Conv2D(filters=120, kernel_size=(5, 5), padding="same", activation="relu")
#         self.flatten = Flatten()
#         self.fc2 = Dense(units=84, activation="relu")
#         self.fc3 = Dense(units=10, activation="softmax")

#     def call(self, input_tensor):
#         # don't create layers here, need to create the layers in initializer,
#         # otherwise you will get the tf.Variable can only be created once error
#         conv1 = self.conv1(input_tensor)
#         maxpool1 = self.max_pool2x2(conv1)
#         conv2 = self.conv2(maxpool1)
#         maxpool2 = self.max_pool2x2(conv2)
#         conv3 = self.conv3(maxpool2)
#         flatten = self.flatten(conv3)
#         fc2 = self.fc2(flatten)
#         fc3 = self.fc3(fc2)

#         return fc3


# input_layer = Input(
#     shape=(
#         32,
#         32,
#         3,
#     )
# )
# x = LeNet5()(input_layer)

# model = Model(inputs=input_layer, outputs=x)

# print(model.summary(expand_nested=True))
