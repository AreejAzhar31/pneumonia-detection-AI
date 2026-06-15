# 🫁 Pneumonia Detection AI

> **AI-powered chest X-ray classifier built with EfficientNetB3 and Flask**  
> Detects pneumonia from chest X-ray images and generates structured clinical recommendation reports.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔍 What It Does

Upload a chest X-ray image and the system:

1. **Classifies** it as `NORMAL` or `PNEUMONIA` using a fine-tuned EfficientNetB3 CNN
2. **Returns confidence scores** for each class
3. **Generates a clinical report** with observed symptoms, recommended actions, and important cautions
4. **Saves the report** as both `.txt` and `.json` in the `reports/` folder

The web interface lets you drag-and-drop X-rays directly in the browser with real-time results.

---

## 🖥️ Demo

```
Image: patient_xray.jpeg
──────────────────────────────────────────────
Diagnosis   : PNEUMONIA
Confidence  : 94.3%

NORMAL      5.7%   ██
PNEUMONIA   94.3%  ██████████████████████████████
```

---

## 🗂️ Project Structure

```
pneumonia_ai/
│
├── app.py                  ← CLI entry point (preprocess / train / predict)
├── server.py               ← Flask web server (serves UI + /predict API)
├── test_model.py           ← Quick CLI batch tester (no server needed)
├── pneumonia_ai.html       ← Frontend interface
├── requirements.txt        ← Python dependencies
│
├── modules/
│   ├── preprocess.py       ← Module 1: Sort & resize dataset images
│   ├── train.py            ← Module 2: Train EfficientNetB3 (transfer learning + fine-tuning)
│   ├── inference.py        ← Module 3: Load model & classify X-ray
│   └── recommendations.py ← Module 4: Generate diagnosis report
│
├── model/                  ← best_model.h5 saved here after training (gitignored)
├── logs/                   ← Training curves & confusion matrix (gitignored)
├── reports/                ← Generated reports (gitignored)
└── data/                   ← Dataset goes here (gitignored — see Setup Step 4)
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- `pip`
- (Optional but recommended) NVIDIA GPU with CUDA for faster training

---

### Step 1 — Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-ai.git
cd pneumonia-ai
```

---

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|----------|---------|
| Windows  | `.venv\Scripts\activate` |
| Mac/Linux | `source .venv/bin/activate` |

You should see `(.venv)` at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs TensorFlow, Flask, Pillow, scikit-learn, matplotlib, seaborn, and more. May take a few minutes.

---

### Step 4 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click **Download** (requires a free Kaggle account)
3. Extract the zip and place the contents so your folder looks like:

```
data/raw/chest_xray/
    train/
        NORMAL/       ← JPEG chest X-rays
        PNEUMONIA/    ← JPEG chest X-rays
    test/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
```

---

### Step 5 — Preprocess the Dataset

```bash
python app.py preprocess
```

Resizes all images to 224×224 and organises them into `data/train/`, `data/val/`, and `data/test/`.

---

### Step 6 — Train the Model

```bash
python app.py train
```

Trains EfficientNetB3 in two phases:
- **Phase 1:** Classification head only (base frozen)
- **Phase 2:** Fine-tuning of the top 30 layers

Training takes **20–60 minutes** depending on your hardware (GPU strongly recommended).  
Outputs:
- `model/best_model.h5` — best checkpoint
- `model/pneumonia_model_final.h5` — final weights
- `logs/training_curves.png` — accuracy & loss plots
- `logs/confusion_matrix.png` — validation confusion matrix

---

### Step 7 — Run the Web Interface

```bash
python server.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

Upload any chest X-ray JPEG and click **Analyze X-ray** to get a prediction with a full clinical report.

---

### Step 8 — (Optional) CLI Batch Testing

Test individual images or whole folders without starting the server:

```bash
# Single image
python test_model.py path/to/xray.jpeg

# Entire folder
python test_model.py data/test/PNEUMONIA/

# Multiple files
python test_model.py xray1.jpg xray2.jpg xray3.jpg
```

Or via `app.py`:

```bash
python app.py predict data/test/NORMAL/IM-0001-0001.jpeg
```

---

## 🧠 Model Architecture

| Component | Detail |
|-----------|--------|
| Base model | EfficientNetB3 (pretrained on ImageNet) |
| Input size | 224 × 224 × 3 |
| Head layers | GlobalAveragePooling → BN → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax(2) |
| Phase 1 LR | 1e-4 (head only, base frozen) |
| Phase 2 LR | 1e-5 (top 30 base layers unfrozen) |
| Loss | Categorical cross-entropy |
| Output classes | `NORMAL`, `PNEUMONIA` |

**Data augmentation** (training only): horizontal flip, rotation ±15°, zoom ±15%, width/height shift ±10%, brightness variation ±20%.

---

## 📊 Output Labels

| Label | Meaning | Dataset Source |
|-------|---------|----------------|
| `NORMAL` | No pneumonia detected | `NORMAL/` folder |
| `PNEUMONIA` | Pneumonia detected | `PNEUMONIA/` folder |

---

## 📋 Generated Report Example

Each prediction saves a report to `reports/`. Sample output:

```
══════════════════════════════════════════════════════════
  PNEUMONIA DETECTION — RECOMMENDATION REPORT
══════════════════════════════════════════════════════════
  Date / Time  : 2026-06-07  13:40:42
  Diagnosis    : Pneumonia Detected
  Confidence   : 94.3%
══════════════════════════════════════════════════════════

  DESCRIPTION
  The chest X-ray shows signs consistent with pneumonia.
  Medical evaluation is strongly recommended.

  OBSERVED / EXPECTED SYMPTOMS
    • Fever and chills
    • Productive cough
    • Shortness of breath

  RECOMMENDED MEDICATIONS & ACTIONS
    • Consult a physician immediately
    • Antipyretics (e.g., Paracetamol) for fever under physician guidance
    • Supplemental oxygen if SpO2 is below 94%

  ⚠  CAUTIONS
    • This result is AI-generated — confirm with a licensed physician
    • Do NOT delay evaluation

══════════════════════════════════════════════════════════
  DISCLAIMER: For informational purposes only.
  Always consult a qualified physician.
══════════════════════════════════════════════════════════
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'PIL'` | `pip install Pillow` |
| `ModuleNotFoundError: No module named 'flask'` | `pip install flask flask-cors` |
| `Model not found at 'model/best_model.h5'` | Run `python app.py train` first |
| `No module named 'tensorflow'` | `pip install tensorflow` |
| Server starts but browser shows nothing | Ensure `pneumonia_ai.html` is in the same folder as `server.py` |
| `(.venv)` not appearing | Re-run the activate command (Step 2) |

---

## ⚠️ Medical Disclaimer

> This tool is intended for **educational and research purposes only**.  
> It is **not a medical device** and must not be used as a substitute for professional medical diagnosis.  
> Always consult a qualified physician before making any clinical decisions.

---


