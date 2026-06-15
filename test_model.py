"""
test_model.py — Quick model tester
====================================
Drop this file in your Pneumonia_AI folder and run:

    python test_model.py xray1.jpg xray2.jpg ...

Or test a whole folder:

    python test_model.py chest_xray/test/PNEUMONIA/

No server needed. Just Python + the model file.
"""

import os
import sys
import numpy as np
from PIL import Image

# ── Config (must match training) ──────────────────────────
IMG_SIZE    = 224
MODEL_PATH  = os.path.join('model', 'best_model.h5')
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def load_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  Model not found at '{MODEL_PATH}'")
        print("    Make sure you're running this from inside the Pneumonia_AI folder.")
        sys.exit(1)
    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    # Print output shape so we can verify it matches CLASS_NAMES
    out_shape = model.output_shape[-1]
    print(f"✓ Model loaded — output classes: {out_shape}")
    if out_shape != len(CLASS_NAMES):
        print(f"\n⚠️  WARNING: Model has {out_shape} output neurons but CLASS_NAMES has {len(CLASS_NAMES)}.")
        print("   This means class labels will be WRONG. Retrain the model or fix CLASS_NAMES.")
    return model

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_one(model, image_path):
    arr   = preprocess(image_path)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), {c: float(p) for c, p in zip(CLASS_NAMES, probs)}

def collect_images(paths):
    """Accept files or folders."""
    images = []
    exts   = ('.jpg', '.jpeg', '.png')
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(exts):
                    images.append(os.path.join(p, f))
        elif os.path.isfile(p) and p.lower().endswith(exts):
            images.append(p)
        else:
            print(f"⚠️  Skipping (not found or unsupported): {p}")
    return images

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    model  = load_model()
    images = collect_images(sys.argv[1:])

    if not images:
        print("No valid image files found.")
        sys.exit(1)

    print(f"\nTesting {len(images)} image(s)...\n")
    print(f"{'Image':<40} {'Result':<12} {'Confidence':>10}   Probabilities")
    print("─" * 90)

    normal_count    = 0
    pneumonia_count = 0

    for path in images:
        try:
            label, conf, probs = predict_one(model, path)
            prob_str = "  ".join(f"{c}:{p*100:.1f}%" for c, p in probs.items())
            flag = "🔴 PNEUMONIA" if label == "PNEUMONIA" else "🟢 NORMAL"
            print(f"{os.path.basename(path):<40} {flag:<20} {conf*100:>6.1f}%   {prob_str}")
            if label == "PNEUMONIA":
                pneumonia_count += 1
            else:
                normal_count += 1
        except Exception as e:
            print(f"{os.path.basename(path):<40} ❌ ERROR: {e}")

    print("─" * 90)
    print(f"\nSummary: {pneumonia_count} PNEUMONIA  |  {normal_count} NORMAL  |  {len(images)} total")

if __name__ == '__main__':
    main()
