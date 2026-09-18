# NeuroVision AI

**Federated Learning + Explainable AI for Privacy-Preserving Brain Tumor Diagnosis using MRI Images**

This is a complete, runnable recreation of the system described in the paper  
*“NeuroVision AI: Federated Learning and Explainable AI for Privacy-Preserving Brain Tumor Diagnosis Using MRI Images”*.

---

## Features (matching the paper)

| Component                    | Implementation                                      |
|-----------------------------|-----------------------------------------------------|
| Privacy-preserving training | Federated Learning with **FedAvg**                  |
| Multiple institutions       | 4 simulated clients (hospitals)                     |
| Deep Learning model         | CNN (TensorFlow / Keras)                            |
| Explainability              | **Grad-CAM** heatmaps                               |
| Classes                     | Glioma, Meningioma, Pituitary, No Tumor             |
| Web application             | FastAPI backend + ready-to-use HTML frontend        |
| Dataset                     | Official Kaggle Brain Tumor MRI Dataset             |

---

## Project Structure

```
NeuroVision-AI/
├── config.py                 # All hyperparameters & paths
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                  # Full dataset (after download)
│   ├── clients/              # Split per institution
│   └── processed/
├── models/
│   ├── cnn_model.py          # CNN architecture
│   ├── federated.py          # FedAvg implementation
│   └── gradcam.py            # Grad-CAM explainability
├── training/
│   ├── train_federated.py    # Main FL training script
│   └── evaluate.py           # Metrics (Acc, P, R, F1, ROC-AUC)
├── api/
│   └── main.py               # FastAPI prediction + Grad-CAM server
├── frontend/
│   └── index.html            # Single-page UI (no build step)
├── scripts/
│   ├── download_dataset.py   # Kaggle download
│   └── prepare_clients.py    # Partition into institutions
└── saved_models/
    └── global_model.h5       # Created after training
```

---

## 1. Setup

```bash
cd NeuroVision-AI

# Create virtual environment (recommended)
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Download Dataset from Kaggle

The project uses the popular **Brain Tumor MRI Dataset**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  
(4 classes: glioma / meningioma / pituitary / notumor ≈ 7 000 images)

**One-time Kaggle setup:**

1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`
3. Run:

```bash
python scripts/download_dataset.py
```

This downloads, extracts and organises the images under `data/raw/`.

---

## 3. Create Federated Clients (Institutions)

```bash
python scripts/prepare_clients.py
```

Splits the data into 4 client folders (`data/clients/client_0` … `client_3`).

---

## 4. Train with Federated Learning (FedAvg)

```bash
python training/train_federated.py
```

- Each client trains locally on its own MRI data
- Only model weights are sent to the server
- Server performs **Federated Averaging**
- Final global model is saved to `saved_models/global_model.h5`

(You can change number of rounds / clients in `config.py`)

---

## 5. Evaluate

```bash
python training/evaluate.py
```

Prints Accuracy, Precision, Recall, F1-Score and Macro ROC-AUC.

---

## 6. Run the Web Application

**Terminal 1 – API**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 – Frontend**

Simply open the file in a browser:

```bash
# Linux / macOS
xdg-open frontend/index.html
# or just double-click frontend/index.html
```

Or serve it:

```bash
cd frontend
python -m http.server 3000
# then open http://localhost:3000
```

Upload any MRI image → receive predicted class + Grad-CAM heatmap.

---

## Expected Workflow (Interview Explanation)

1. **Data stays local** – each hospital (client) never shares MRI images.
2. **Local training** – CNN is trained for a few epochs on the client’s data.
3. **Parameter aggregation** – only weights are averaged (FedAvg).
4. **Global model** – improved model is redistributed.
5. **Inference + Explainability** – Grad-CAM highlights the regions that drove the prediction, increasing clinical trust.
6. **Web UI** – doctors can upload a scan and instantly see the diagnosis + explanation.

---

## Configuration

Edit `config.py` to change:

- `NUM_CLIENTS`, `NUM_ROUNDS`, `LOCAL_EPOCHS`
- `IMG_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`
- Class names / paths

---

## Notes / Limitations of the Original Paper

- Exact CNN architecture, hyper-parameters and quantitative results were **not** provided in the paper → we supply a clean, standard CNN that works with Grad-CAM.
- BraTS is primarily a **segmentation** dataset; the classification task is performed on the public 4-class Kaggle MRI dataset named above.
- The React frontend mentioned in the paper is replaced by a zero-build HTML/JS frontend that talks to the same FastAPI backend (functionality identical).

---

## License & Academic Use

This implementation is for educational and research purposes.  
Always cite the original paper and the Kaggle dataset when using the code.
