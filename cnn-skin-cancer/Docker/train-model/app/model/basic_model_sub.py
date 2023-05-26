from keras.models import Model
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau


params = {
    "num_classes": 2,
    "input_shape": (224, 224, 3),
    "activation": "relu",
    "kernel_initializer_glob": "glorot_uniform",
    "kernel_initializer_norm": "normal",
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
    "validation_split": 0.2,
    "epochs": 2,
    "batch_size": 64,
}

# https://towardsdatascience.com/tensorflow-class-inheritance-beautiful-code-59d2eb7cdfce
# https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e



class BasicNet(Model):
    def __init__(self, params:dict):
        super(BasicNet, self).__init__()
        #creating layers in initializer
        self.conv_input = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="Same",
            #input_shape=params.get("input_shape"),
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
        self.dpo = layers.Dropout(0.25),
        self.flatten = layers.Flatten(),
        self.fc2 = layers.Dense(
            128, activation=params.get("activation"), kernel_initializer=params.get("kernel_initializer_norm")
        )
        self.fc3 = layers.Dense(params.get("num_classes"), activation="softmax")

    # TODO: typehint
    def call(self, input_tensor):
        # don't create layers here, need to create the layers in initializer,
        # otherwise you will get the tf.Variable can only be created once error
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

    def build_graph(self, raw_shape):
        x = layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))

input_layer = params.get("input_shape")
model = BasicNet(params)
print(model.summary(expand_nested=True))

model.compile(optimizer=params.get("optimizer"), loss=params.get("loss"), metrics=params.get("metrics"))
#model.summary()

# The first call to the `cm` will create the weights
# y = cm(tf.ones(shape=(0,*raw_input))) 

model.build_graph(input_layer).summary()

#model = Model(inputs=input_layer, outputs=x)

