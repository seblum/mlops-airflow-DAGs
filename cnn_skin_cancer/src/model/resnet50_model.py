import tensorflow.keras.applications.resnet50 as rs50
from keras.models import Model
from keras.optimizers import Adam


class ResNet50:
    def __init__(self, model_params: dict):
        self.model = rs50.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=model_params.get("input_shape"),
            pooling=model_params.get("pooling"),
            classes=model_params.get("num_classes"),
        )

    def call(self) -> Model:
        return self.model
