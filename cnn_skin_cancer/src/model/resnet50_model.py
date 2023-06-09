import tensorflow.keras.applications.resnet50 as rs50
from keras.models import Model


class ResNet50:
    """ResNet50 is a wrapper class for the ResNet-50 model from Keras applications.

    Args:
        model_params (dict): A dictionary containing the model parameters.

    Attributes:
        model (keras.models.Model): The ResNet-50 model.

    Methods:
        call() -> keras.models.Model:
            Returns the ResNet-50 model.

    """

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
        """Returns the ResNet-50 model.

        Returns:
            keras.models.Model: The ResNet-50 model.

        """
        return self.model
