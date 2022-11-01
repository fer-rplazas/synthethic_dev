import argparse
import os
import warnings
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from joblib import Parallel, delayed
from pytorch_lightning.callbacks import ModelCheckpoint

from configs import configs
from tools.data_generation import DataGenerator
from tools.nn_training import Module, ModuleAR
from tools.svm import SVMClassifier
from tools.utils import Dataset

Fs = 2048  # sampling frequency in [Hz]
T = 1200  # Total simulated dataset length in [s]
window_length = 0.5  # Window length in [s]
hop_size = 0.4  # Hop size in [s]. Determines overlap between windows

from torch.cuda import device_count, is_available

DEVICE = "gpu" if is_available() else "cpu"


def score_module(module, train_dataloader, valid_dataloader, accelerator, device):

    trainer = pl.Trainer(
        logger=True,
        accelerator=accelerator,
        devices=device,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="valid/bal_acc",
                save_top_k=1,
            )
        ],
        max_epochs=150,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        trainer.fit(module, train_dataloader, valid_dataloader)
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        train_score = trainer.validate(module, train_dataloader)[0]["valid/bal_acc"]
        valid_score = trainer.validate(module, valid_dataloader)[0]["valid/bal_acc"]

    return train_score, valid_score


def score_dataset(
    file,
    cfg,
    snr,
    save_to_file=True,
    accelerator: str = "gpu",
    device: list[int] | str = [0],
):

    scores = pd.Series(
        index=[
            "snr",
            "svm_train",
            "svm_valid",
            "cnn1d_train",
            "cnn1d_valid",
            "cnn2d_train",
            "cnn2d_valid",
            "ar_train",
            "ar_valid",
        ]
    )
    scores["snr"] = snr

    data = DataGenerator(cfg, snr, Fs=Fs, T=T)
    dataset = Dataset(
        data.signal,
        data.label,
        Fs,
        tf_transform=True,
        ar_len=10,
        window_length=window_length,
        hop_size=hop_size,
    )

    # SVM:
    cls = SVMClassifier(
        dataset.X_features_train_scaled,
        dataset.X_features_valid_scaled,
        dataset.y_train,
        dataset.y_valid,
    )

    scores["svm_train"], scores["svm_valid"] = cls.classify()

    # 1D-CNN:
    module = Module.with_defaults_1d(data.signal.shape[0])
    scores["cnn1d_train"], scores["cnn1d_valid"] = score_module(
        module,
        dataset.train_dataloader(),
        dataset.valid_dataloader(),
        accelerator,
        device,
    )

    # 2d-CNN:
    module = Module.with_defaults_2d(data.signal.shape[0])
    scores["cnn2d_train"], scores["cnn2d_valid"] = score_module(
        module,
        dataset.train_tf_dataloader(),
        dataset.valid_tf_dataloader(),
        accelerator,
        device,
    )

    # AR Model:
    module = ModuleAR.with_defaults(data.signal.shape[0], dataset.X_features.shape[1])
    scores["ar_train"], scores["ar_valid"] = score_module(
        module,
        dataset.train_ar_dataloader(),
        dataset.valid_ar_dataloader(),
        accelerator,
        device,
    )

    if save_to_file:

        sleep(np.random.rand(1)[0] * 4)  # Avoid write conflicts from multiprocessing

        df = pd.read_csv(file, index_col=0)
        df = pd.concat((df, scores.to_frame().T), ignore_index=True)
        df.to_csv(file)

    return scores


def prepare_runs(name):

    filepath = Path("_temp") / Path(name).with_suffix(".csv")

    if not os.path.exists(filepath):
        df = pd.DataFrame(
            columns=[
                "snr",
                "svm_train",
                "svm_valid",
                "cnn1d_train",
                "cnn1d_valid",
                "cnn2d_train",
                "cnn2d_valid",
                "ar_train",
                "ar_valid",
            ]
        )

        df.to_csv(filepath)

    return filepath


def sweep_snr(
    name: os.PathLike, cfg: dict, n_jobs: int = 4, accelerator: str | int = "gpu"
):
    """Prepares database, configures accelerators, and launches parallelized sweep across a range or snrs for the configuration specified in `cfg`"""
    filepath = prepare_runs(name)

    snrs = np.repeat(np.arange(0.2, 2, 0.05), 5)

    if accelerator == "cpu":
        accelerator = "cpu"
        devices = ["auto" for _ in range(snrs.size)]
    elif accelerator == "gpu":  # Spread jobs across available gpus
        accelerator = "gpu"
        devices = np.tile(
            list(range(device_count())), (snrs.size // device_count()) + 1
        )
        devices = devices[: snrs.size]
        devices = [[int(x)] for x in devices]
    elif isinstance(accelerator, int):
        accelerator = "gpu"
        devices = [[accelerator] for _ in range(snrs.size)]
    else:
        raise ValueError("`accelerator` not recognized.")

    assert len(devices) == snrs.size, "Problem deciding on gpus for sweep"

    out = Parallel(n_jobs=n_jobs)(
        delayed(score_dataset)(
            filepath, cfg, snr=snr, accelerator=accelerator, device=device
        )
        for snr, device in zip(snrs, devices)
    )


def is_int(el: str):
    """Determines whether a str can be converted to int"""
    try:
        int(el)
        return True
    except ValueError:
        return False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--cfg_name", required=False, default=None)
    parser.add_argument("--n_jobs", required=False, default=4, type=int)
    parser.add_argument("--accelerator", required=False, default="gpu", type=str)

    args = parser.parse_args()

    # Choose compatible accelerator:
    if DEVICE == "cpu":
        if is_int(args.accelerator) or args.accelerator == "gpu":
            warnings.warn("No GPU detected on your system -- using cpu.", UserWarning)
            args.accelerator = "cpu"
    else:
        args.accelerator = (
            int(args.accelerator) if is_int(args.accelerator) else args.accelerator
        )

    args.cfg_name = args.name if args.cfg_name is None else args.cfg_name

    sweep_snr(
        args.name,
        configs[args.cfg_name],
        n_jobs=args.n_jobs,
        accelerator=args.accelerator,
    )


if __name__ == "__main__":
    main()
