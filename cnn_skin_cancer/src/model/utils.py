from enum import Enum

from basic_model_sub import BasicNet
from crossval_model import *
from keras.models import Model
from resnet50_model_sub import ResNet50


class Model_Class(Enum):
    """This enum includes different models."""

    Basic = "Basic"
    CrossVal = "CrossVal"
    ResNet50 = "ResNet50"


def get_model(model_name: str, model_params: dict) -> Model:

    # TODO update python version
    match model_name:
        case Model_Class.Basic:
            model = BasicNet(model_params)
            # print(model.summary(expand_nested=True))
            model.compile(
                optimizer=model_params.get("optimizer"),
                loss=model_params.get("loss"),
                metrics=model_params.get("metrics"),
            )
            print(model.build_graph(model_params).summary())
            return model

        case Model_Class.CrossVal:
            pass
        case Model_Class.ResNet50:
            model = ResNet50(model_params).call()
            model.compile(
                optimizer=model_params.get("optimizer"),
                loss=model_params.get("loss"),
                metrics=model_params.get("metrics"),
            )
            print(model.summary(expand_nested=True))
            return model