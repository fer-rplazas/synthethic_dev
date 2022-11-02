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
        optimizer_hparams,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.BCEWithLogitsLoss()

    @classmethod
    def with_defaults_1d(cls, n_in: int):
        return cls(
            "cnn1d", {"n_channels": n_in}, "Adam", {"lr": 1e-3, "weight_decay": 1e-4}
        )

    @classmethod
    def with_defaults_2d(cls, n_in: int):
        return cls(
            "cnn2d", {"n_channels": n_in}, "Adam", {"lr": 1e-3, "weight_decay": 1e0}
        )

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        else:
            raise ValueError("optimizer_name not recognized")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, _):

        x, y = batch

        logits = self.model(x).squeeze()
        loss = self.loss_module(logits, y.float())

        preds = torch.sigmoid(logits) > 0.5
        bal_acc = balanced_accuracy_score(
            y.detach().cpu().numpy().squeeze(), preds.detach().cpu().numpy().squeeze()
        )

        return {"loss": loss, "bal_acc": bal_acc}

    def training_epoch_end(self, outs):

        losses = np.array([out["loss"].detach().cpu().numpy() for out in outs])
        bal_accs = np.array([out["bal_acc"] for out in outs])

        self.log("train/bal_acc", np.mean(bal_accs))
        self.log("train/loss", np.mean(losses))

    def validation_step(self, batch, _):

        x, y = batch

        logits = self.model(x).squeeze()
        loss = self.loss_module(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        bal_acc = balanced_accuracy_score(
            y.detach().cpu().numpy().squeeze(), preds.detach().cpu().numpy().squeeze()
        )

        return {"loss": loss, "bal_acc": bal_acc}

    def validation_epoch_end(self, outs):

        losses = np.array([out["loss"].detach().cpu().numpy() for out in outs])
        bal_accs = np.array([out["bal_acc"] for out in outs])
        self.score = (1 - np.mean(losses)) + np.mean(bal_accs)
        self.log("valid/bal_acc", np.mean(bal_accs))
        self.log("valid/loss", np.mean(losses))
        self.log("valid/score", self.score)


class ModuleAR(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.BCEWithLogitsLoss()

    @classmethod
    def with_defaults(cls, n_in: int, n_feats: int):
        return cls(
            "ARConvs",
            {"n_channels": n_in, "n_feats": n_feats},
            "Adam",
            {"lr": 1e-3, "weight_decay": 1e-4},
        )

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        else:
            raise ValueError("optimizer_name not recognized")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, _):

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

    def training_epoch_end(self, outs):

        losses = np.array([out["loss"].detach().cpu().numpy() for out in outs])
        bal_accs = np.array([out["bal_acc"] for out in outs])

        self.log("train/bal_acc", np.mean(bal_accs))
        self.log("train/loss", np.mean(losses))

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

    def validation_epoch_end(self, outs):

        losses = np.array([out["loss"].detach().cpu().numpy() for out in outs])
        bal_accs = np.array([out["bal_acc"] for out in outs])
        self.running_score = (1 - np.mean(losses)) + np.mean(bal_accs)

        self.log_dict(
            {
                "valid/bal_acc": np.mean(bal_accs),
                "valid/loss": np.mean(losses),
                "valid/score": self.running_score,
            }
        )
