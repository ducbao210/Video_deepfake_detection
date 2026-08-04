#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
from tqdm import tqdm

DATA_DIR = Path("data/raw")

REAL_DIR = DATA_DIR / "DFD_original_sequences"
FAKE_DIR = DATA_DIR / "DFD_manipulated_sequences"


def normalize_dataset_structure(data_dir: Path):
    """
    Normalize the dataset directory structure after extraction.

    - Rename original dataset
    - Flatten mainpulated dataset
    """
    old_real = data_dir / "DFD_original sequences"
    new_real = data_dir / "DFD_original_sequences"

    if old_real.exists():
        if new_real.exists():
            shutil.rmtree(old_real)
        else:
            old_real.rename(new_real)

    outer_fake = data_dir / "DFD_manipulated_sequences"
    inner_fake = outer_fake / "DFD_manipulated_sequences"

    if inner_fake.exists():
        for item in inner_fake.iterdir():
            destination = outer_fake / item.name

            if destination.exists():
                continue

            shutil.move(str(item), str(destination))

        inner_fake.rmdir()

    # remove tmp51ews_lu file in original dataset
    tmp_file = new_real / "tmp51ews_lu"
    if tmp_file.exists() and tmp_file.is_file():
        tmp_file.unlink()


try:
    if REAL_DIR.exists() and FAKE_DIR.exists():
        print("Dataset already exists. Skipping dowload.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tqdm(
            total=1,
            desc="Dowloading dataset",
            bar_format="{desc} [{elapsed}]",
        ) as pbar:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset",
                    "-p",
                    str(DATA_DIR),
                    "--unzip",
                ],
                check=True,
            )

            pbar.update(1)

        zip_path = DATA_DIR / "deep-fake-detection-dfd-entire-original-dataset.zip"
        if zip_path.exists():
            zip_path.unlink()

        normalize_dataset_structure(DATA_DIR)
        print("Dataset downloaded and normalized successfully.")

except FileNotFoundError:
    print("The 'kaggle' command was not found.")
    print("Please install the Kaggle CLI and ensure it is in your PATH.")

except subprocess.CalledProcessError as e:
    print("The Kaggle CLI encountered an error.")
    print(e)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
