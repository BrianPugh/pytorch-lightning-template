from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path = Path(path)
        # self.fns = [f.name for f in (self.path / "images").glob("*.jpg")]
        self.fns = ["foo", "bar", "baz"]
        self.transform = transform

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        out_dict = {
            "image": None,
            "gt": None,
        }

        fn = self.fns[idx]
        image_fn = self.path / "images" / (fn + ".jpg")
        gt_fn = self.path / "gt" / (fn + ".png")

        image = cv2.cvtColor(cv2.imread(str(image_fn)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_fn))

        if self.transform is not None:
            image = self.transform(image)

        out_dict["image"] = image
        out_dict["gt"] = gt
        out_dict["fn"] = fn  # Useful debugging metadata

        return out_dict
