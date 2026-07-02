#!/usr/bin/env python3
"""
predict.py
----------
Usage:
    python predict.py some_image.jpg
    -> prints a single number from 0 to 1
       0 = real photo, 1 = photo of a screen (recapture)

    python predict.py some_image.jpg --label
    -> also prints REAL / SCREEN using the model's optimized operating
       threshold (not a hardcoded 0.5)

Loads model.pkl (produced by train.py) and scores one image. Works with
both a single calibrated classifier and an EnsembleModel (soft-vote /
weighted fusion of several calibrated classifiers) - both expose the same
predict_proba(X) interface.
"""

import argparse
import sys
import time

import joblib
import numpy as np

from extract_features import extract_features
# needed so joblib can unpickle an EnsembleModel if that's what was saved
from ensemble import EnsembleModel  # noqa: F401


def load_model(path="model.pkl"):
    bundle = joblib.load(path)
    return bundle


def predict(image_path, bundle):
    feats = extract_features(image_path)
    feature_indices = bundle.get("feature_indices")
    if feature_indices is not None:
        feats = feats[feature_indices]
    feats = feats.reshape(1, -1)
    feats_s = bundle["scaler"].transform(feats)
    prob_screen = bundle["model"].predict_proba(feats_s)[0, 1]
    return float(prob_screen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to image file")
    ap.add_argument("--model", default="model.pkl")
    ap.add_argument("--timeit", action="store_true",
                     help="print latency in ms to stderr")
    ap.add_argument("--label", action="store_true",
                     help="also print REAL/SCREEN using the model's "
                          "optimized operating threshold")
    args = ap.parse_args()

    bundle = load_model(args.model)

    t0 = time.time()
    score = predict(args.image, bundle)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"{score:.4f}")
    if args.label:
        threshold = bundle.get("operating_threshold", 0.5)
        label = "SCREEN" if score >= threshold else "REAL"
        print(f"[label: {label}  (threshold={threshold:.2f})]", file=sys.stderr)
    if args.timeit:
        print(f"[latency: {elapsed_ms:.1f} ms]", file=sys.stderr)


if __name__ == "__main__":
    main()