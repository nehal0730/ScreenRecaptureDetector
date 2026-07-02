"""
hybrid_cnn.py  (OPTIONAL / EXPERIMENTAL - not enabled by default)
-------------------------------------------------------------------
Adds a late-fusion hybrid: probability from the handcrafted-feature model
(model.pkl) averaged with a probability from a small frozen pretrained CNN
(MobileNetV3-Small) + a lightweight logistic-regression head trained on
your data.

WHY THIS IS OPTIONAL, NOT THE DEFAULT:
  - Your primary dataset is ~50-60 unique photos per class. Even after
    augmentation to ~500, there are still only ~50-60 *independent* real-
    world scenes. A CNN head trained on that is at real risk of memorizing
    the small set of source photos (via their augmented copies) rather
    than learning a generalizable rule - the augmented copies are
    correlated, not independent evidence.
  - MobileNetV3-Small is still ~9MB and needs a torch/torchvision runtime,
    versus ~200KB-1MB and pure-numpy/opencv for the handcrafted model -
    a real cost in the "small, fast, cheap" criteria the brief asks for.
  - I could not test this script end-to-end in the sandbox I built this
    in (no network access there to download ImageNet-pretrained weights,
    and torch/torchvision aren't installed) - it's written carefully and
    should run correctly on your machine, but verify its output before
    relying on it. Run predict.py's plain handcrafted model as your
    primary/submitted solution; treat this as an experiment to report on
    in the "hybrid" discussion, not as your main deliverable.

Usage (after `pip install torch torchvision`):
    python hybrid_cnn.py train --data_dir data_augmented --group_cv
    python hybrid_cnn.py predict some_image.jpg
"""

import argparse
import glob
import os
import re
import time

import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from extract_features import extract_features

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
CNN_MODEL_PATH = "hybrid_cnn_head.pkl"


def _require_torch():
    if not HAS_TORCH:
        raise SystemExit(
            "torch/torchvision/pillow not installed. "
            "Run: pip install torch torchvision pillow"
        )


def get_cnn_embedder():
    """MobileNetV3-Small, pretrained on ImageNet, frozen, used purely as a
    fixed feature extractor (no fine-tuning - fine-tuning ~500 correlated
    images is a strong overfitting risk)."""
    _require_torch()
    net = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    net.classifier = nn.Identity()  # strip the classification head -> embedding
    net.eval()
    return net


_preprocess = None


def _get_preprocess():
    global _preprocess
    if _preprocess is None:
        _preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return _preprocess


def cnn_embedding(image_path, net):
    _require_torch()
    img = Image.open(image_path).convert("RGB")
    x = _get_preprocess()(img).unsqueeze(0)
    with torch.no_grad():
        emb = net(x)
    return emb.squeeze(0).numpy()


def load_paths(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def source_group_id(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"__aug\d+$", "", base)


def train(data_dir, group_cv):
    _require_torch()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    net = get_cnn_embedder()

    X, y, groups = [], [], []
    for label, folder_name in ((0, "real"), (1, "screen")):
        paths = load_paths(os.path.join(data_dir, folder_name))
        print(f"  {folder_name}: {len(paths)} images")
        for p in paths:
            emb = cnn_embedding(p, net)
            X.append(emb)
            y.append(label)
            groups.append(source_group_id(p))

    X, y, groups = np.array(X), np.array(y), np.array(groups)
    n_unique = len(set(groups))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    head = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)

    n_splits = max(2, min(5, n_unique))
    if group_cv:
        cv = GroupKFold(n_splits=n_splits)
        preds = cross_val_predict(head, Xs, y, cv=cv, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        preds = cross_val_predict(head, Xs, y, cv=cv)

    acc = accuracy_score(y, preds)
    print(f"CNN-embedding + LogisticRegression head, "
          f"{'group-' if group_cv else ''}CV accuracy: {acc*100:.1f}%")
    print("Compare this to model.pkl's handcrafted-feature accuracy printed "
          "by train.py. Only use the hybrid if this meaningfully beats it "
          "AND the latency/size cost (see note.md) is acceptable to you.")

    head.fit(Xs, y)
    joblib.dump({"scaler": scaler, "head": head, "cv_accuracy": acc}, CNN_MODEL_PATH)
    print(f"Saved CNN head to {CNN_MODEL_PATH}")


def predict(image_path, handcrafted_model_path="model.pkl"):
    _require_torch()
    hc_bundle = joblib.load(handcrafted_model_path)
    hc_scaler, hc_model = hc_bundle["scaler"], hc_bundle["model"]
    feats = extract_features(image_path).reshape(1, -1)
    hc_prob = hc_model.predict_proba(hc_scaler.transform(feats))[0, 1]

    cnn_bundle = joblib.load(CNN_MODEL_PATH)
    cnn_scaler, cnn_head = cnn_bundle["scaler"], cnn_bundle["head"]
    net = get_cnn_embedder()
    emb = cnn_embedding(image_path, net).reshape(1, -1)
    cnn_prob = cnn_head.predict_proba(cnn_scaler.transform(emb))[0, 1]

    fused = 0.5 * hc_prob + 0.5 * cnn_prob
    print(f"handcrafted={hc_prob:.4f}  cnn={cnn_prob:.4f}  fused={fused:.4f}")
    return fused


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data_dir", default="data_augmented")
    t.add_argument("--group_cv", action="store_true")

    p = sub.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--model", default="model.pkl")

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data_dir, args.group_cv)
    else:
        predict(args.image, args.model)


if __name__ == "__main__":
    main()