"""
CNN model for brain tumor classification.
Architecture follows the paper description:
multiple convolutional layers + pooling + fully-connected layers,
ReLU activations, Softmax output, Adam optimizer,
categorical cross-entropy loss.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, CHANNELS, NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE, DENSE_UNITS


def build_cnn(input_shape=(*IMG_SIZE, CHANNELS), num_classes=NUM_CLASSES):
    """
    Build a practical CNN that matches the paper's high-level description
    and works well with Grad-CAM (has clear Conv2D layers).
    """
    model = models.Sequential(name="NeuroVision_CNN")

    # Block 1
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4 – last conv used by Grad-CAM
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv4"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(DENSE_UNITS, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model():
    """Factory used by both centralized and federated training."""
    return build_cnn()


if __name__ == "__main__":
    m = get_model()
    m.summary()
