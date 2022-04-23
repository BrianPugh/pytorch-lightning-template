import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Strip non-model-weights from checkpoint."
    )
    parser.add_argument("input", type=Path, help="Checkpoint file to strip.")
    parser.add_argument(
        "--output", "-o", type=Path, help="New checkpoint file to output."
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")

    def remove(key):
        checkpoint.pop(key, None)

    # We could remove other metadata, but they're small and might be useful if
    # trying to figure out where a model came from.
    remove("optimizer_states")

    if args.output is None:
        output = args.input.parent / f"{args.input.stem}-stripped{args.input.suffix}"
    else:
        output = args.output

    torch.save(checkpoint, output)


if __name__ == "__main__":
    main()
