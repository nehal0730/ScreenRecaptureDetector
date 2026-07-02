"""
app.py
------
Local web server for the live camera demo.

    python app.py
    -> open http://127.0.0.1:5000  (or the phone's LAN IP if run on a
       laptop and opened from a phone browser on the same network)

This does NOT reimplement feature extraction in JavaScript - the browser
just captures a frame and POSTs it to /predict, which runs the exact same
extract_features.py + model.pkl your train.py produced. No image is
written to disk or sent anywhere off this machine.
"""

import base64
import os
import time

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from extract_features import extract_features, FEATURE_NAMES
from ensemble import EnsembleModel  # noqa: F401  (needed so joblib can unpickle an EnsembleModel)

MODEL_PATH = "model.pkl"
app = Flask(__name__)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None and os.path.exists(MODEL_PATH):
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# A handful of interpretable features surfaced as live "signal readouts" in
# the UI, purely for explainability. The model itself may rely on a
# different (automatically selected) subset internally - these are just
# for the human watching the demo to see roughly what the detector notices.
# (lo, hi) are rough, illustrative min/max ranges for the 0-1 bar display,
# not calibrated statistics.
DISPLAY_SIGNALS = [
    ("jpeg_blockiness", 0.8, 3.0),
    ("glare_ratio", 0.0, 0.15),
    ("fft_high_freq_ratio", 0.3, 0.7),
    ("lbp_uniform_ratio", 0.4, 0.9),
]


def normalize(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo + 1e-9), 0, 1))


@app.route("/")
def index():
    bundle = get_bundle()
    return render_template("index.html", model_missing=(bundle is None))


@app.route("/status")
def status():
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"ready": False})
    return jsonify({
        "ready": True,
        "model_name": bundle.get("model_name", "unknown"),
        "threshold": bundle.get("operating_threshold", 0.5),
        "cv_accuracy": bundle.get("cv_accuracy"),
    })


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"error": "model.pkl not found - run augment.py then train.py first"}), 400

    data = request.get_json(silent=True) or {}
    data_url = data.get("image", "")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")
    except Exception as e:
        return jsonify({"error": f"bad image payload: {e}"}), 400

    t0 = time.time()
    feats = extract_features(img)

    feature_indices = bundle.get("feature_indices")
    model_feats = feats[feature_indices] if feature_indices is not None else feats
    model_feats = model_feats.reshape(1, -1)
    feats_s = bundle["scaler"].transform(model_feats)
    prob_screen = float(bundle["model"].predict_proba(feats_s)[0, 1])
    elapsed_ms = (time.time() - t0) * 1000

    threshold = bundle.get("operating_threshold", 0.5)
    label = "REAL" if prob_screen >= threshold else "SCREEN"

    signals = []
    for name, lo, hi in DISPLAY_SIGNALS:
        idx = FEATURE_NAMES.index(name)
        raw = float(feats[idx])
        signals.append({
            "name": name,
            "raw": raw,
            "normalized": normalize(raw, lo, hi),
        })

    return jsonify({
        "score": prob_screen,
        "label": label,
        "threshold": threshold,
        "latency_ms": elapsed_ms,
        "signals": signals,
    })


if __name__ == "__main__":
    print("Starting demo server: http://127.0.0.1:5000")
    print("(to open from a phone on the same WiFi, use this machine's LAN IP instead of 127.0.0.1)")
    app.run(debug=False, host="0.0.0.0", port=5000)