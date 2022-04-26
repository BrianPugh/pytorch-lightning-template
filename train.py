import argparse
import importlib
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.strategies import BaguaStrategy
from torch.utils.data import DataLoader
from torchvision import transforms

from app.datasets import MyDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MyModel")
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=80_000)
    parser.add_argument(
        "--fine-tune",
        type=Path,
        help="Restore weights directly via model, not Trainer.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_steps": args.iterations,
        "callbacks": [
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    }
    dataloader_kwargs = {}
    model_kwargs = {}

    if args.debug:
        trainer_kwargs["detect_anomaly"] = True
        trainer_kwargs["log_every_n_steps"] = 1
    else:
        # Assumes GPU training
        trainer_kwargs["strategy"] = BaguaStrategy(algorithm="gradient_allreduce")
        trainer_kwargs["gpus"] = -1
        trainer_kwargs["log_every_n_steps"] = args.log_freq
        trainer_kwargs["precision"] = 16
        trainer_kwargs["amp_backend"] = "native"
        dataloader_kwargs["batch_size"] = args.batch_size
        dataloader_kwargs["num_workers"] = 16
        dataloader_kwargs["pin_memory"] = True

    Model = getattr(importlib.import_module("app.models"), args.model)

    if args.fine_tune:
        trainer_kwargs.pop("resume_from_checkpoint", None)
        Model = Model.load_from_checkpoint
        model_kwargs["checkpoint_path"] = str(args.fine_tune)
        # model_kwargs["strict"] = False

    model = Model(**model_kwargs)

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = MyDataset(
        args.data,
        transform=train_transforms,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
