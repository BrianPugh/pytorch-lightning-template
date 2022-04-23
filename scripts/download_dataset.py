import argparse
from pathlib import Path

from torchvision.datasets.utils import download_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/download"),
        help="Folder to download raw data to.",
    )
    args = parser.parse_args()

    urls = [
        "http://website-hosting-some-dataset.com/file.zip",
    ]
    for url in urls:
        print(f"Downloading {url}")
        download_url(url, args.output)

    print("Done! Now run:")
    print("    python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
