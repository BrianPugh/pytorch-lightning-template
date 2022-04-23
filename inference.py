"""Minimal script to run trained model.
"""
import argparse
import importlib
from pathlib import Path

import cv2
from torchvision import transforms


def resize_shortest(image, size):
    h, w = image.shape[:2]
    if h < w:
        target_h = size
        target_w = int(round((target_h / h) * w))
    else:
        target_w = size
        target_h = int(round((target_w / w) * h))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MyModel")
    parser.add_argument(
        "--weights",
        "-w",
        type=Path,
        required=True,
        help="Path to weights (.pth/.ckpt) file.",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Image to run inference on."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("out.png"),
        help="Output prediction file.",
    )
    args = parser.parse_args()

    Model = getattr(importlib.import_module("app.models"), args.model)
    model = Model.load_from_checkpoint(args.weights)

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f'"{args.input}" does not exist.')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_shortest(image, 480)

    image_tensor = test_transforms(image)[0]

    pred = model(image_tensor)
    pred = pred[0, 0].detach().cpu().numpy()

    args.output.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(args.output), image)


if __name__ == "__main__":
    main()
