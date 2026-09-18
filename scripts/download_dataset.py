"""
Download the official Brain Tumor MRI Dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes : glioma, meningioma, pituitary, notumor

HOW TO USE:
1. Go to https://www.kaggle.com/settings → API → Create New Token
2. Place the downloaded kaggle.json in ~/.kaggle/kaggle.json
   (or set environment variables KAGGLE_USERNAME and KAGGLE_KEY)
3. Run:  python scripts/download_dataset.py
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, KAGGLE_DATASET, CLASS_NAMES


def setup_kaggle():
    """Check that Kaggle credentials exist."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("=" * 60)
        print("Kaggle credentials not found!")
        print()
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' under API section")
        print("3. Save the file as: ~/.kaggle/kaggle.json")
        print("   (or export KAGGLE_USERNAME and KAGGLE_KEY)")
        print("=" * 60)
        sys.exit(1)

    # Ensure correct permissions
    os.chmod(kaggle_json, 0o600)
    print("[OK] Kaggle credentials found.")


def download_and_extract():
    """Download the dataset using the Kaggle API and organise folders."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    download_path = Path(RAW_DIR).parent / "kaggle_download"
    download_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {KAGGLE_DATASET} ...")
    print("       This may take a few minutes (≈150 MB).")

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(download_path),
        unzip=True,
        quiet=False
    )

    print("[OK] Download finished. Organising folders...")

    # The dataset usually has Training/ and Testing/ subfolders
    # We merge them into a single class-based structure for federated splits.
    target_classes = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary",
        "notumor": "notumor",
        "no_tumor": "notumor",
        "no tumor": "notumor",
        "glioma_tumor": "glioma",
        "meningioma_tumor": "meningioma",
        "pituitary_tumor": "pituitary",
    }

    # Clear previous raw data
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)

    image_count = {cls: 0 for cls in CLASS_NAMES}

    for root, dirs, files in os.walk(download_path):
        folder_name = Path(root).name.lower().replace(" ", "_")
        mapped = None
        for key, val in target_classes.items():
            if key in folder_name:
                mapped = val
                break
        if mapped is None:
            continue

        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                src = os.path.join(root, f)
                dst_name = f"{image_count[mapped]:05d}_{f}"
                dst = os.path.join(RAW_DIR, mapped, dst_name)
                shutil.copy2(src, dst)
                image_count[mapped] += 1

    print("\n[OK] Dataset organised under data/raw/")
    for cls, cnt in image_count.items():
        print(f"     {cls:12s}: {cnt} images")
    total = sum(image_count.values())
    print(f"     TOTAL       : {total} images")

    # Clean temporary download
    shutil.rmtree(download_path, ignore_errors=True)
    print("\n[DONE] Ready for client partitioning.")


if __name__ == "__main__":
    setup_kaggle()
    download_and_extract()
