"""Download the cat-faces dataset used by ddpm.py via kagglehub and copy it
into a local ./data/cats folder (ddpm.py's default --data_dir).

Usage:
    python -m src.download_data [--target-dir ./data/cats]
"""
import argparse
import os
import shutil

import kagglehub


def main():
    parser = argparse.ArgumentParser(description="Download and stage the cat-faces dataset")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=os.path.join("data", "cats"),
        help="where to copy the downloaded dataset (default: ./data/cats)",
    )
    args = parser.parse_args()

    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("borhanitrash/cat-dataset")
    print("Path to dataset files:", path)

    os.makedirs(args.target_dir, exist_ok=True)
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(args.target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f"Done. Dataset copied to: {args.target_dir}")


if __name__ == "__main__":
    main()
