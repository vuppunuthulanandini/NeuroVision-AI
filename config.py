"""
NeuroVision AI - Configuration
All hyperparameters and paths used across the project.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CLIENTS_DIR = os.path.join(DATA_DIR, "clients")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CLIENTS_DIR, exist_ok=True)

# ============================================================
# DATASET (Kaggle: masoudnickparvar/brain-tumor-mri-dataset)
# Classes exactly as described in the paper
# ============================================================
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASS_NAMES)

# Kaggle dataset identifier
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

# ============================================================
# IMAGE SETTINGS
# ============================================================
IMG_SIZE = (224, 224)          # Standard for CNN + Grad-CAM
CHANNELS = 3

# ============================================================
# FEDERATED LEARNING (as described in the paper)
# ============================================================
NUM_CLIENTS = 4                # Simulated healthcare institutions
CLIENT_FRACTION = 1.0          # All clients participate every round
LOCAL_EPOCHS = 3               # Local training epochs per round
BATCH_SIZE = 32
NUM_ROUNDS = 8                 # Federated communication rounds
LEARNING_RATE = 1e-4

# ============================================================
# TRAINING
# ============================================================
SEED = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================
# MODEL
# ============================================================
DROPOUT_RATE = 0.5
DENSE_UNITS = 128
