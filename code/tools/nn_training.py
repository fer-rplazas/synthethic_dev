from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from .arch import create_model


class Module(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
        use_mixup: bool = True,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.BCEWithLogitsLoss()

        self.use_mixup = use_mixup

    @classmethod
    def with_defaults_1d(cls, n_in: int):
        # return cls(
        #     "cnn1d", {"n_channels": n_in}, "Adam", {"lr": 1e-3, "weight_decay": 1e-4}
        # )
        return cls(
            "cnn1d",
            {"n_channels": n_in, "compress": True, "depth": 8},
            "Adam",
            {"lr": 1e-3, "weight_decay": 1e-8},
        )

    @classmethod
    def with_defaults_fft(cls, n_in: int, n_samples: int):
        return cls(
            "dotfft",
            {"n_channels": n_in, "n_samples": n_samples, "n_hidden": 10, "depth": 10},
            "Adam",
            {"lr": 1e-3, "weight_decay": 1e-5},
        )

    @classmethod
    def with_defaults_2d(cls, n_in: int):
        return cls(
            "cnn2d", {"n_channels": n_in}, "Adam", {"lr": 1e-3, "weight_decay": 1e0}
        )

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        comp_params, other_params = [], []

        # TODO: Determine whether to use different slope for Compressor
        for name, param in self.model.named_parameters():
            # if name.split(".")[-1] == "slope":
            #     comp_params.append(param)
            # else:
            other_params.append(param)

        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(
                [
                    {"params": comp_params, "lr": 1e-1, "weight_decay": 1e-8},
                    {"params": other_params},
                ],
                **self.hparams.optimizer_hparams
            )
        else:
            raise ValueError("optimizer_name not recognized")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, _):

        x, y = batch

        logits = self.model(x).squeeze()

        if self.use_mixup:
            logits_mixup = logits.clone()
            y_mixup = y.clone()
            if len(logits) % 2 != 0:
                logits_mixup = logits_mixup[:-1]
                y_mixup = y_mixup[:-1]

            n = logits_mixup.shape[0]
            lambdas = torch.rand(n // 2).to(logits.device)

            logits_mixup = (
                lambdas * logits_mixup[: n // 2]
                + (1 - lambdas) * logits_mixup[n // 2 :]
            )
            y_mixup = lambdas * y_mixup[: n // 2] + (1 - lambdas) * y_mixup[n // 2 :]

            loss = self.loss_module(logits_mixup, y_mixup.float())
        else:
            loss = self.loss_module(logits, y.float())

        preds = torch.sigmoid(logits) > 0.5
        bal_acc = balanced_accuracy_score(
            y.detach().cpu().numpy().squeeze(), preds.detach().cpu().numpy().squeeze()
        )

        return {"loss": loss, "bal_acc": bal_acc}

    def training_epoch_end(self, metrics):

        loss = np.array(
            [metric["loss"].detach().cpu().numpy() for metric in metrics]
        ).mean()
        bal_acc = np.array([metric["bal_acc"] for metric in metrics]).mean()

        self.log("train/bal_acc", bal_acc)
        self.log("train/loss", loss)

    def validation_step(self, batch, _):

        x, y = batch

        logits = self.model(x).squeeze()
        loss = self.loss_module(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        bal_acc = balanced_accuracy_score(
            y.detach().cpu().numpy().squeeze(), preds.detach().cpu().numpy().squeeze()
        )

        return {"loss": loss, "bal_acc": bal_acc}

    def validation_epoch_end(self, metrics):

        loss = np.array(
            [metric["loss"].detach().cpu().numpy() for metric in metrics]
        ).mean()
        bal_acc = np.array([metric["bal_acc"] for metric in metrics]).mean()
        self.score = (1 - loss) + bal_acc

        self.log("valid/bal_acc", bal_acc)
        self.log("valid/loss", loss)
        self.log("valid/score", self.score)


class ModuleAR(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
        use_mixup: bool = True,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.BCEWithLogitsLoss()

        self.use_mixup = use_mixup

    @classmethod
    def with_defaults(cls, n_in: int, n_feats: int):
        return cls(
            "ARConvs",
            {"n_channels": n_in, "n_feats": n_feats},
            "Adam",
            {"lr": 1e-2, "weight_decay": 1e-8},
        )

    @classmethod
    def with_defaults_ensemble(cls, n_in: int, n_feats: int, n_samples: int):
        return cls(
            "EnsembleAR",
            {"n_channels": n_in, "n_feats": n_feats, "n_samples": n_samples},
            "Adam",
            {"lr": 1e-2, "weight_decay": 1e-8},
        )

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        comp_params, other_params = [], []

        # TODO: Determine whether to use different slope for Compressor
        for name, param in self.model.named_parameters():
            # if name.split(".")[-1] == "slope":
            #     comp_params.append(param)
            # else:
            other_params.append(param)

        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(
                [
                    {"params": comp_params, "lr": 1e-1, "weight_decay": 1e-8},
                    {"params": other_params},
                ],
                **self.hparams.optimizer_hparams
            )
        else:
            raise ValueError("optimizer_name not recognized")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, _):

        # Unpack sequence data in batch:
        signals = torch.stack(
            [el[0] for el in batch]
        ).float()  # (n_seq, n_batch, n_chan, n_times)
        feats = torch.stack([el[1] for el in batch]).float()  # (n_seq, n_batch, n_feat)
        ys = torch.stack([el[2] for el in batch]).float()  # (n_seq, n_batch)

        # Get logits for each sequence step:
        logits = self.model(signals, feats)  # (n_seq, n_batch)

        if self.use_mixup:
            logits_mixup = torch.flatten(logits).clone()
            y_mixup = torch.flatten(ys).clone()
            if len(logits) % 2 != 0:
                logits_mixup = logits_mixup[:-1]
                y_mixup = y_mixup[:-1]

            n = logits_mixup.shape[0]
            lambdas = torch.rand(n // 2).to(logits.device)

            logits_mixup = (
                lambdas * logits_mixup[: n // 2]
                + (1 - lambdas) * logits_mixup[n // 2 :]
            )
            y_mixup = lambdas * y_mixup[: n // 2] + (1 - lambdas) * y_mixup[n // 2 :]

            loss = self.loss_module(logits_mixup, y_mixup)
        else:
            loss = self.loss_module(torch.flatten(logits), torch.flatten(ys))

        # Compute metrics based only on last sequence element:
        preds = torch.sigmoid(logits[-1, :]) > 0.5
        bal_acc = balanced_accuracy_score(
            ys[-1, :].detach().cpu().numpy().squeeze(),
            preds.detach().cpu().numpy().squeeze().astype(float),
        )

        return {"loss": loss, "bal_acc": bal_acc}

    def training_epoch_end(self, metrics: list[dict[str, Any]]):

        loss = np.array(
            [metric["loss"].detach().cpu().numpy() for metric in metrics]
        ).mean()
        bal_acc = np.array([metric["bal_acc"] for metric in metrics]).mean()

        self.log("train/bal_acc", bal_acc)
        self.log("train/loss", loss)

    def validation_step(self, batch, _):

        signals = torch.stack(
            [el[0] for el in batch]
        ).float()  # (n_seq, n_batch, n_chan, n_times)
        feats = torch.stack([el[1] for el in batch]).float()  # (n_seq, n_batch, n_feat)
        ys = torch.stack([el[2] for el in batch]).float()  # (n_seq, n_batch)

        logits = self.model(signals, feats)  # (n_seq, n_batch)
        loss = self.loss_module(torch.flatten(logits), torch.flatten(ys))
        preds = torch.sigmoid(logits[-1, :]) > 0.5
        bal_acc = balanced_accuracy_score(
            ys[-1, :].detach().cpu().numpy().squeeze(),
            preds.detach().cpu().numpy().squeeze().astype(float),
        )

        return {"loss": loss, "bal_acc": bal_acc}

    def validation_epoch_end(self, metrics: list[dict[str, Any]]):

        loss = np.array(
            [metric["loss"].detach().cpu().numpy() for metric in metrics]
        ).mean()
        bal_acc = np.array([metric["bal_acc"] for metric in metrics]).mean()
        self.score = (1 - loss) + bal_acc

        self.log_dict(
            {
                "valid/bal_acc": bal_acc,
                "valid/loss": loss,
                "valid/score": self.score,
            }
        )
