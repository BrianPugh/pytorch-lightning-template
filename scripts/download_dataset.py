import argparse
from pathlib import Path

import lox
from torchvision.datasets.utils import download_url

download_url = lox.thread(download_url)


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
        download_url.scatter(url, args.output)
    download_url.gather(tqdm=True)

    print("Done! Now run:")
    print("    python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
