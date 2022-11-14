from .armodel import ARModel, EnsembleAR
from .cnn1d import CNN1d
from .cnn2d import resnet
from .dotfft import ComplexFFTNet

model_dict = {
    "cnn1d": CNN1d,
    "ARConvs": ARModel,
    "cnn2d": resnet,
    "dotfft": ComplexFFTNet,
    "EnsembleAR": EnsembleAR,
}


def create_model(model_name: str, model_hparams: dict):
    return model_dict[model_name](**model_hparams)
