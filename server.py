# ============================================================
#  server.py — Flask API Server
#  Pneumonia Detection AI Project
#
#  Install deps:  pip install flask flask-cors
#  Run:           python server.py
#  Open:          http://localhost:5000
# ============================================================

import os
import io
import base64
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

from modules.inference       import load_model, CLASS_NAMES, to_recommendation_label
from modules.recommendations import get_recommendation, save_report, save_report_json

app = Flask(__name__, static_folder='.')
CORS(app)

print("\n[Server] Loading model at startup...")
try:
    MODEL = load_model()
    print(f"[Server] ✓ Model ready. Classes: {CLASS_NAMES}\n")
except FileNotFoundError as e:
    print(f"[Server] ✗ {e}")
    MODEL = None

IMG_SIZE = 224

def decode_image(data_url: str) -> np.ndarray:
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict_route():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 503

    data = request.get_json(force=True)
    if 'image' not in data:
        return jsonify({'error': 'No image provided.'}), 400

    try:
        img_array  = decode_image(data['image'])
        probs      = MODEL.predict(img_array, verbose=0)[0]

        # ✅ Uses CLASS_NAMES from inference.py: ['NORMAL', 'PNEUMONIA']
        pred_index = int(np.argmax(probs))
        label      = CLASS_NAMES[pred_index]
        confidence = float(probs[pred_index])
        all_probs  = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}

        # ✅ Convert to recommendation key ('normal' or 'pneumonia')
        rec_label = to_recommendation_label(label)
        report, meta = get_recommendation(rec_label, confidence)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_report(report,    filename=f'report_{timestamp}.txt')
        save_report_json(meta, filename=f'report_{timestamp}.json')

        return jsonify({
            'label':      rec_label,
            'confidence': confidence,
            'allProbs':   all_probs,
            'report':     report
        })

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Inference failed. Check server logs.'}), 500


@app.route('/')
def index():
    return send_from_directory('.', 'pneumonia_ai.html')


if __name__ == '__main__':
    print("="*50)
    print("  Pneumonia AI — Server starting")
    print("  Open: http://localhost:5000")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=False)