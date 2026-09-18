"""
Entry point for Federated Learning training.
Run after:
  1. python scripts/download_dataset.py
  2. python scripts/prepare_clients.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.federated import train_federated


if __name__ == "__main__":
    print("Starting NeuroVision AI Federated Training...")
    model, history = train_federated()
    print("\nTraining history (round → avg accuracy):")
    for r, acc in zip(history["round"], history["avg_client_accuracy"]):
        print(f"  Round {r}: {acc:.4f}")
    print("\nDone. Global model is ready for inference & Grad-CAM.")
