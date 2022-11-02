from .cnn2d import resnet
from .cnn1d import CNN1d
from .armodel import ARModel


model_dict = {"cnn1d": CNN1d, "ARConvs": ARModel, "cnn2d": resnet}


def create_model(model_name: str, model_hparams: dict):
    return model_dict[model_name](**model_hparams)
