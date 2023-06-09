import tensorflow as tf
from keras import layers
from keras.models import Model
from tensorflow import keras


class BasicNet(Model):
    """BasicNet is a custom neural network model derived from the Keras `Model` class.

    Args:
        params (dict): A dictionary containing the model parameters.

    Attributes:
        conv_input (keras.layers.Conv2D): Convolutional layer for input processing.
        max_pool2x2 (keras.layers.MaxPooling2D): Max pooling layer.
        conv_hidden (keras.layers.Conv2D): Convolutional layer for hidden processing.
        dpo (keras.layers.Dropout): Dropout layer for regularization.
        flatten (keras.layers.Flatten): Flatten layer.
        fc2 (keras.layers.Dense): Fully connected layer.
        fc3 (keras.layers.Dense): Output layer.

    Methods:
        call(input_tensor: tf.Tensor) -> tf.Tensor:
            Forward pass of the model.

        build_graph(model_params: dict) -> Model:
            Build the complete model graph.

    """

    def __init__(self, params: dict):
        super(BasicNet, self).__init__()
        # creating layers in initializer
        self.conv_input = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="Same",
            # input_shape=params.get("input_shape"),
            activation=params.get("activation"),
            kernel_initializer=params.get("kernel_initializer_glob"),
        )
        self.max_pool2x2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_hidden = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="Same",
            activation=params.get("activation"),
            kernel_initializer=params.get("kernel_initializer_glob"),
        )
        self.dpo = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.fc2 = layers.Dense(
            128,
            activation=params.get("activation"),
            kernel_initializer=params.get("kernel_initializer_norm"),
        )
        self.fc3 = layers.Dense(params.get("num_classes"), activation="softmax")

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            input_tensor (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.

        """
        conv1 = self.conv_input(input_tensor)
        maxpool1 = self.max_pool2x2(conv1)
        conv2 = self.conv_hidden(maxpool1)
        dpo1 = self.dpo(conv2)
        conv3 = self.conv_hidden(dpo1)
        maxpool2 = self.max_pool2x2(conv3)
        dpo2 = self.dpo(maxpool2)
        flatten = self.flatten(dpo2)
        fc2 = self.fc2(flatten)
        fc3 = self.fc3(fc2)
        return fc3

    def build_graph(self, model_params: dict) -> Model:
        """Build the complete model graph.

        Args:
            model_params (dict): A dictionary containing the model parameters.

        Returns:
            Model: The compiled Keras model.

        """
        raw_shape = model_params.get("input_shape")
        x = layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))
