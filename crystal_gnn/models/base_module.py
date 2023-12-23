from abc import ABCMeta, abstractmethod

from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanAbsoluteError, R2Score
from transformers import get_constant_schedule_with_warmup


class BaseModule(LightningModule, metaclass=ABCMeta):
    def __init__(self, _config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(_config)
        self.num_classes = _config["num_classes"]
        self.readout_dim = self.num_classes if self.num_classes != 2 else 1
        # log
        if self.num_classes > 1:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        else:
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score()
        # optimizer
        self.optimizer = _config["optimizer"]
        self.lr = _config["lr"]
        self.weight_decay = _config["weight_decay"]
        self.scheduler = _config["scheduler"]

    @abstractmethod
    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self,
        batch: Union[Data, Batch],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # get logits
        logits = self.forward(batch)
        logits = logits.squeeze()
        # get target
        target = batch["target"].to(logits.dtype)
        # get train_mean and train_std
        train_mean = float(batch["train_mean"][0] if "train_mean" in batch else 0)
        train_std = float(batch["train_std"][0] if "train_std" in batch else 1)
        # calculate loss
        if self.num_classes == 1:
            target = (target - train_mean) / train_std  # encode
        loss = self._calculate_loss(logits, target)
        if self.num_classes == 1:
            logits = (logits * train_std) + train_mean  # decode
            target = (target * train_std) + train_mean  # decode
        # log metrics
        self._log_metrics(
            logits,
            target,
            "train",
            loss=loss,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: Union[Data, Batch],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # get logits
        logits = self.forward(batch)
        logits = logits.squeeze()
        # get target
        target = batch["target"].to(logits.dtype)
        # get train_mean and train_std
        train_mean = float(batch["train_mean"][0] if "train_mean" in batch else 0)
        train_std = float(batch["train_std"][0] if "train_std" in batch else 1)
        # calculate loss
        if self.num_classes == 1:
            target = (target - train_mean) / train_std  # encode
        loss = self._calculate_loss(logits, target)
        if self.num_classes == 1:
            logits = (logits * train_std) + train_mean  # decode
            target = (target * train_std) + train_mean  # decode
        # log metrics
        self._log_metrics(
            logits,
            target,
            "val",
            loss=loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(
        self,
        batch: Union[Data, Batch],
        batch_idx,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # get logits
        logits = self.forward(batch)
        logits = logits.squeeze()
        # get target
        target = batch["target"].to(logits.dtype)
        # get train_mean and train_std
        train_mean = float(batch["train_mean"][0] if "train_mean" in batch else 0)
        train_std = float(batch["train_std"][0] if "train_std" in batch else 1)
        # calculate loss
        if self.num_classes == 1:
            target = (target - train_mean) / train_std  # encode
        loss = self._calculate_loss(logits, target)
        if self.num_classes == 1:
            logits = (logits * train_std) + train_mean  # decode
            target = (target * train_std) + train_mean  # decode
        # log metrics
        self._log_metrics(
            logits,
            target,
            "test",
            loss=loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def predict_step(
        self,
        batch: Union[Data, Batch],
        batch_idx,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        logits = self.forward(batch)
        logits = logits.squeeze()
        # get train_mean and train_std
        train_mean = float(batch["train_mean"][0] if "train_mean" in batch else 0)
        train_std = float(batch["train_std"][0] if "train_std" in batch else 1)
        if self.num_classes == 1:
            logits = (logits * train_std) + train_mean  # decode
        elif self.num_classes == 2:
            logits = torch.sigmoid(logits)
            logits = logits > 0.5
        else:
            logits = logits.argmax(dim=1)

        return logits.squeeze()

    def configure_optimizers(self) -> Dict[str, Any]:
        return self._set_configure_optimizers()

    # TODO: deprecated
    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _calculate_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.num_classes == 1:
            loss = F.mse_loss(logits, target)
        elif self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(logits, target)
            # the bce includes a sigmoid fuction
        else:
            loss = F.cross_entropy(logits, target)
        return loss

    def _log_metrics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        split: str,
        loss: Optional[torch.Tensor] = None,
        on_step: bool = False,
        on_epoch: bool = False,
    ) -> None:
        self.log(
            f"{split}/loss",
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )
        if self.num_classes == 1:
            self.log(
                f"{split}/mae",
                self.mae(logits.squeeze(), target),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=self.hparams.batch_size,
            )
            self.log(
                f"{split}/r2",
                self.r2(logits.squeeze(), target),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=self.hparams.batch_size,
            )
        elif self.num_classes == 2:
            logits = F.sigmoid(logits)
            logits = logits > 0.5
            self.log(
                f"{split}/accuracy",
                self.accuracy(logits.squeeze(), target),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=self.hparams.batch_size,
            )
        else:
            logits = logits.argmax(dim=1)
            self.log(
                f"{split}/accuracy",
                self.accuracy(logits.squeeze(), target),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=self.hparams.batch_size,
            )

    def _set_configure_optimizers(self):
        lr = self.lr
        weight_decay = self.weight_decay
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        # get max_steps
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        # set scheduler
        if self.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif self.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min"
            )
        elif self.scheduler == "linear_decay":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, max_steps)
        elif self.scheduler == "constant_with_warmup":
            warmup_step = int(max_steps * 0.05)
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_step
            )
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning rate",
            "monitor": "val/loss",
        }

        return ([optimizer], [lr_scheduler])
