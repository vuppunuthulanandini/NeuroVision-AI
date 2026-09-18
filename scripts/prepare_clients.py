"""
Partition the raw dataset into multiple simulated healthcare institutions
(clients) for Federated Learning, exactly as described in the paper.
"""

import os
import sys
import shutil
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, CLIENTS_DIR, NUM_CLIENTS, CLASS_NAMES, SEED


def split_into_clients():
    random.seed(SEED)
    np.random.seed(SEED)

    # Clean previous client folders
    if os.path.exists(CLIENTS_DIR):
        shutil.rmtree(CLIENTS_DIR)

    for c in range(NUM_CLIENTS):
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(CLIENTS_DIR, f"client_{c}", cls), exist_ok=True)

    print(f"[INFO] Splitting data into {NUM_CLIENTS} clients (institutions)...")

    for cls in CLASS_NAMES:
        src_dir = Path(RAW_DIR) / cls
        if not src_dir.exists():
            print(f"[WARN] Missing class folder: {src_dir}")
            continue

        images = list(src_dir.glob("*.*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        random.shuffle(images)

        # Roughly equal non-IID friendly split
        chunks = np.array_split(images, NUM_CLIENTS)

        for c, chunk in enumerate(chunks):
            for img in chunk:
                dst = Path(CLIENTS_DIR) / f"client_{c}" / cls / img.name
                shutil.copy2(img, dst)

        print(f"  {cls:12s}: {len(images)} images → distributed")

    # Summary
    print("\n[OK] Client distribution:")
    for c in range(NUM_CLIENTS):
        total = 0
        counts = {}
        for cls in CLASS_NAMES:
            n = len(list((Path(CLIENTS_DIR) / f"client_{c}" / cls).glob("*.*")))
            counts[cls] = n
            total += n
        print(f"  Client {c}: {total:4d} images  {counts}")

    print("\n[DONE] Clients ready for Federated Learning.")


if __name__ == "__main__":
    split_into_clients()
