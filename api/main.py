"""
FastAPI backend for NeuroVision AI.
- Accepts MRI image upload
- Returns predicted class + confidence
- Returns Grad-CAM heatmap (base64 PNG)

Matches the web application described in the paper
(React frontend + FastAPI backend).
"""

import os
import io
import sys
import base64
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, CLASS_NAMES, IMG_SIZE
from models.gradcam import (
    make_gradcam_heatmap,
    overlay_gradcam,
    get_last_conv_layer_name,
)

app = FastAPI(
    title="NeuroVision AI",
    description="Privacy-preserving Brain Tumor Diagnosis with Federated Learning & Grad-CAM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model immediately when the file is imported
model = None
last_conv_layer = None

model_path = os.path.join(MODEL_DIR, "global_model.h5")
print(f"[INFO] Looking for model at: {model_path}")

if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        last_conv_layer = get_last_conv_layer_name(model)
        print(f"[OK] Model loaded successfully. Last conv layer = {last_conv_layer}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = None
else:
    print(f"[WARN] Model file not found at {model_path}")


def preprocess_bytes(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return arr, batch


@app.get("/")
def root():
    return {
        "message": "NeuroVision AI API is running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img, batch = preprocess_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    preds = model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    heatmap = make_gradcam_heatmap(
        batch, model, last_conv_layer, pred_index=class_idx
    )
    overlay = overlay_gradcam((img * 255).astype("uint8"), heatmap)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", overlay_bgr)
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(
        {
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(preds[i]), 4)
                for i in range(len(CLASS_NAMES))
            },
            "gradcam_heatmap": heatmap_b64,
        }
    )