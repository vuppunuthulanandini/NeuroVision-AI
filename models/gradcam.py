"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
as described in the paper for visual explanations of MRI predictions.
"""
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None
):
    """
    Generate a Grad-CAM heatmap for a trained Keras model.

    Parameters
    ----------
    img_array : numpy array
        Shape: (1, H, W, C), normalized to [0, 1].

    model : tf.keras.Model
        Trained NeuroVision CNN.

    last_conv_layer_name : str
        Name of the convolutional layer used for Grad-CAM.

    pred_index : int, optional
        Class index for which Grad-CAM is generated.
    """

    # ============================================================
    # STEP 1: Create a fresh Functional input
    # ============================================================

    inputs = tf.keras.Input(
        shape=model.input_shape[1:],
        name="gradcam_input"
    )

    x = inputs

    conv_output = None

    # ============================================================
    # STEP 2: Rebuild the model graph using the SAME layers
    # and SAME trained weights
    # ============================================================

    for layer in model.layers:

        # Sequential model contains an InputLayer.
        # We already created our own Input above.
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(x)

        # Save output of conv4
        if layer.name == last_conv_layer_name:
            conv_output = x

    # Make sure the requested convolutional layer exists
    if conv_output is None:
        raise ValueError(
            f"Could not find convolutional layer: "
            f"{last_conv_layer_name}"
        )

    # x is now the final prediction output
    predictions = x

    # ============================================================
    # STEP 3: Create Functional Grad-CAM model
    # ============================================================

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            conv_output,
            predictions
        ]
    )

    # ============================================================
    # STEP 4: Forward pass + GradientTape
    # ============================================================

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        # If class index wasn't provided,
        # select the class with highest probability.
        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        # Get prediction score for selected class
        class_channel = predictions[:, pred_index]

    # ============================================================
    # STEP 5: Calculate gradients
    # ============================================================

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:
        raise ValueError(
            "Grad-CAM gradients are None. "
            "The convolutional layer is not connected "
            "to the prediction output."
        )

    # ============================================================
    # STEP 6: Average gradients
    # ============================================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # ============================================================
    # STEP 7: Weight feature maps using gradients
    # ============================================================

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    # ============================================================
    # STEP 8: ReLU
    # ============================================================

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # ============================================================
    # STEP 9: Normalize heatmap
    # ============================================================

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


def overlay_gradcam(
    img,
    heatmap,
    alpha=0.45
):
    """
    Overlay Grad-CAM heatmap on the original MRI image.

    Parameters
    ----------
    img : numpy array
        RGB uint8 image [0, 255].

    heatmap : numpy array
        Grad-CAM heatmap.

    alpha : float
        Heatmap transparency.
    """

    # Resize heatmap to original image dimensions
    heatmap = cv2.resize(
        heatmap,
        (
            img.shape[1],
            img.shape[0]
        )
    )

    # Convert [0, 1] → [0, 255]
    heatmap = np.uint8(
        255 * heatmap
    )

    # Apply OpenCV color map
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR → convert to RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Blend original image and heatmap
    superimposed = (
        heatmap_color * alpha
        + img * (1 - alpha)
    )

    # Keep valid pixel range
    superimposed = np.clip(
        superimposed,
        0,
        255
    ).astype("uint8")

    return superimposed


def get_last_conv_layer_name(model):
    """
    Return the name of the last Conv2D layer.
    """

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            return layer.name

    raise ValueError(
        "No Conv2D layer found in the model"
    )