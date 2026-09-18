"""
Evaluate the global model on a held-out test set (from raw data)
and print Accuracy, Precision, Recall, F1-Score, ROC-AUC
as mentioned in the paper.
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, MODEL_DIR, CLASS_NAMES, IMG_SIZE, BATCH_SIZE


def evaluate():
    model_path = os.path.join(MODEL_DIR, "global_model.h5")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("Run training first: python training/train_federated.py")
        return

    model = load_model(model_path)
    print(f"[OK] Loaded model from {model_path}")

    # Use the full raw dataset with a validation split as a proxy test set
    # (In a real multi-hospital setting you would keep a separate test set)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    test_gen = datagen.flow_from_directory(
        RAW_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        subset="validation",
    )

    print("\nRunning predictions...")
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy : {acc:.4f}")

    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        print(f"Macro ROC-AUC    : {roc:.4f}")
    except Exception as e:
        print(f"ROC-AUC could not be computed: {e}")

    print("=" * 60)


if __name__ == "__main__":
    evaluate()
