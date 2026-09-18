"""
MRI preprocessing utilities.
Operations described in the paper: resize, normalize, augmentation
(rotation, flipping, zooming).
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, CLASS_NAMES, CLIENTS_DIR


def load_and_preprocess_image(path, target_size=IMG_SIZE):
    """Load a single MRI image, convert to RGB, resize and normalize."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img


def create_augmentation_generator():
    """
    Data augmentation as mentioned in the paper:
    rotation, flipping, zooming.
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1.0 / 255.0,
    )


def create_plain_generator():
    """Generator without augmentation (for validation / test)."""
    return ImageDataGenerator(rescale=1.0 / 255.0)


def prepare_client_data(client_id, batch_size=32, augment=True, shuffle=True):
    """
    Create a Keras ImageDataGenerator flow for one client (institution).
    """
    client_path = os.path.join(CLIENTS_DIR, f"client_{client_id}")
    if not os.path.exists(client_path):
        raise FileNotFoundError(f"Client folder not found: {client_path}")

    datagen = create_augmentation_generator() if augment else create_plain_generator()

    generator = datagen.flow_from_directory(
        client_path,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=shuffle,
        color_mode="rgb",
    )
    return generator


def get_class_indices():
    """Return the mapping used by Keras generators."""
    return {name: idx for idx, name in enumerate(CLASS_NAMES)}
