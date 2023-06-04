from enum import Enum

from keras.models import Model

# for Docker
from model.basic_model import BasicNet
from model.resnet50_model import ResNet50

# for Airflow
# from cnn_skin_cancer.src.model.basic_model import BasicNet
# from cnn_skin_cancer.src.model.resnet50_model import ResNet50

# https://towardsdatascience.com/tensorflow-class-inheritance-beautiful-code-59d2eb7cdfce
# https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e


class Model_Class(Enum):
    """This enum includes different models."""

    Basic = "Basic"
    CrossVal = "CrossVal"
    ResNet50 = "ResNet50"


def get_model(model_name: str, model_params: dict) -> Model:
    # TODO update python version
    print(f"name: {model_name}")
    match model_name:
        case Model_Class.Basic.value:
            print("I am here")
            model = BasicNet(model_params)
            print(model)
            # print(model.summary(expand_nested=True))
            model.compile(
                optimizer=model_params.get("optimizer"),
                loss=model_params.get("loss"),
                metrics=model_params.get("metrics"),
            )
            # TODO: doesnt need to print every time
            print(model.build_graph(model_params).summary())
            return model

        case Model_Class.CrossVal.value:
            pass
        case Model_Class.ResNet50.value:
            model = ResNet50(model_params).call()
            model.compile(
                optimizer=model_params.get("optimizer"),
                loss=model_params.get("loss"),
                metrics=model_params.get("metrics"),
            )
            print(model.summary(expand_nested=True))
            return model
