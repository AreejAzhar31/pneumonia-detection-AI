import os
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Configuration ─────────────────────────────────────────
IMG_SIZE   = 224
MODEL_PATH = os.path.join('model', 'best_model.h5')

# ✅ FIX: Must match train.py — Keras reads folders alphabetically
# chest_xray/train has: NORMAL, PNEUMONIA → indices {'NORMAL': 0, 'PNEUMONIA': 1}
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# ============================================================
#  STEP 1 — Load Saved Model
# ============================================================

def load_model():
    """Load the trained model from Module 2."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Please run modules/train.py (Module 2) first."
        )
    print(f"[Module 3] Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("  ✓ Model loaded successfully.")
    return model

# ============================================================
#  STEP 2 — Preprocess Image  (same pipeline as training)
# ============================================================

def preprocess_image(image_path):
    """
    Resize to 224×224, convert to RGB, normalize to [0,1].
    Exactly mirrors training preprocessing.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)
    return img_array

# ============================================================
#  STEP 3 — Run Prediction
# ============================================================

def predict(model, image_path):
    """
    Run inference on a single X-ray image.
    Returns:
        label      (str)  — 'NORMAL' or 'PNEUMONIA'
        confidence (float) — 0.0 to 1.0
        all_probs  (dict)  — probabilities for both classes
    """
    img_array  = preprocess_image(image_path)
    probs      = model.predict(img_array, verbose=0)[0]   # shape: (2,)

    # ✅ FIX: model has 2 outputs — guard against shape mismatches
    if len(probs) != len(CLASS_NAMES):
        raise ValueError(
            f"Model output has {len(probs)} classes but CLASS_NAMES has "
            f"{len(CLASS_NAMES)}. Retrain or fix CLASS_NAMES."
        )

    pred_index = int(np.argmax(probs))
    label      = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    all_probs  = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    return label, confidence, all_probs

# ============================================================
#  STEP 4 — Display Result
# ============================================================

def display_result(image_path, label, confidence, all_probs):
    """Print a clean formatted result to the terminal."""
    print("\n" + "="*50)
    print("  PNEUMONIA DETECTION RESULT")
    print("="*50)
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Diagnosis : {label.upper()}")
    print(f"  Confidence: {confidence*100:.1f}%")
    print("\n  Class Probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {cls:<12} {prob*100:5.1f}%  {bar}")
    print("="*50)

# ============================================================
#  STEP 5 — Map to recommendation label
# ============================================================

def to_recommendation_label(label):
    """
    ✅ FIX: Convert NORMAL/PNEUMONIA to the label keys used
    by recommendations.py knowledge base.
    """
    mapping = {
        'NORMAL':    'normal',
        'PNEUMONIA': 'pneumonia',
    }
    return mapping.get(label, 'normal')

# ============================================================
#  MAIN — Run inference on a single image
# ============================================================

def run_inference(image_path):
    """Full inference pipeline — load model, predict, display."""
    print("\n[Module 3] Inference & Classification Engine")
    model = load_model()
    print(f"[Module 3] Processing image: {image_path}")
    label, confidence, all_probs = predict(model, image_path)
    display_result(image_path, label, confidence, all_probs)

    # Convert to recommendation label for Module 4
    rec_label = to_recommendation_label(label)
    return rec_label, confidence, all_probs


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python modules/inference.py <path_to_xray_image>")
        print("Example: python modules/inference.py chest_xray/test/PNEUMONIA/sample.jpeg")
    else:
        run_inference(sys.argv[1])