import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/download"),
        help="Downloaded data folder.",
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("data"))
    args = parser.parse_args()

    with zipfile.ZipFile(args.input / "file.zip", "r") as f:
        f.extractall(args.output)


if __name__ == "__main__":
    main()
