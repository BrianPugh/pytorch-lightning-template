import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .common import LoggingMixin


class MyModel(pl.LightningModule, LoggingMixin):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):

        pred = self(batch["image"])

        loss = 0
        loss += F.l1_loss(batch["gt"], pred)

        self.log("train_loss", loss)

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_rgb("image", batch["image"][0])

        return loss

    def configure_optimizers(self):
        max_steps = self.trainer.max_steps
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(max_steps * 0.9),
            ],
            gamma=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
