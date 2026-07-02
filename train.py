"""
train.py
--------
Full pipeline:
  1. Extract features (41-dim) from data_dir/real and data_dir/screen.
  2. Rank feature importance and compare full vs. reduced feature set
     (drop the low-value tail) - keep whichever is smaller without
     costing meaningful accuracy.
  3. Compare classifiers (RandomForest, XGBoost/LightGBM if installed)
     via GroupKFold (grouped by source photo, so augmented copies never
     split across train/test - avoids inflated/dishonest accuracy).
  4. Compare individual models vs. soft-voting vs. accuracy-weighted
     averaging; keep whichever wins.
  5. Optimize the decision threshold (accuracy / F1 / balanced accuracy)
     on out-of-fold probabilities instead of assuming 0.5.
  6. Save error analysis: confusion matrix, ROC curve, PR curve, feature
     importance plot, and copies of misclassified images.
  7. Calibrate probabilities (Platt/sigmoid via CalibratedClassifierCV,
     itself using grouped CV to avoid leakage) and save the final model.

Usage:
    python augment.py --data_dir data --out_dir data_augmented
    python train.py --data_dir data_augmented --group_cv
"""

import argparse
import glob
import os
import re
import shutil
import time

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from extract_features import extract_features, FEATURE_NAMES
from ensemble import EnsembleModel

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_paths(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def augment_image(img):
    """Cheap in-memory augmentations (used only with --augment on raw data;
    prefer augment.py + --group_cv for the full offline pipeline)."""
    out = []
    h, w = img.shape[:2]
    out.append(cv2.flip(img, 1))
    for angle in (-8, 8):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    for alpha, beta in ((1.15, 10), (0.85, -10)):
        out.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))
    ch, cw = int(h * 0.85), int(w * 0.85)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = img[y0:y0 + ch, x0:x0 + cw]
    out.append(cv2.resize(crop, (w, h)))
    return out


def source_group_id(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"__aug\d+$", "", base)


def build_dataset(data_dir, augment=False):
    X, y, groups, paths_out = [], [], [], []
    for label, folder_name in ((0, "real"), (1, "screen")):
        folder = os.path.join(data_dir, folder_name)
        paths = load_paths(folder)
        print(f"  {folder_name}: {len(paths)} images")
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"  [skip, unreadable] {p}")
                continue
            variants = [(img, p)]
            if augment:
                variants += [(v, p) for v in augment_image(img)]
            gid = source_group_id(p)
            for v, src_path in variants:
                feats = extract_features(v)
                X.append(feats)
                y.append(label)
                groups.append(gid)
                paths_out.append(src_path)
    return np.array(X), np.array(y), np.array(groups), paths_out


# --------------------------------------------------------------------------
# Feature importance / selection
# --------------------------------------------------------------------------

def rank_feature_importance(Xs, y):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=7, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(Xs, y)
    order = np.argsort(-rf.feature_importances_)
    return order, rf.feature_importances_


def select_top_features(order, importances, cum_threshold=0.90):
    total = importances.sum() + 1e-12
    cum = np.cumsum(importances[order]) / total
    k = int(np.searchsorted(cum, cum_threshold) + 1)
    k = max(k, 5)  # never go below a small floor
    keep_idx = sorted(order[:k].tolist())
    return keep_idx


# --------------------------------------------------------------------------
# Classifier + ensemble comparison
# --------------------------------------------------------------------------

def get_candidate_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    else:
        print("  [info] xgboost not installed - skipping "
              "(pip install xgboost to include it)")
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        print("  [info] lightgbm not installed - skipping "
              "(pip install lightgbm to include it)")
    return models


def cv_probs(model, Xs, y, groups, use_group_cv, n_splits):
    if use_group_cv:
        cv = GroupKFold(n_splits=n_splits)
        probs = cross_val_predict(model, Xs, y, cv=cv, groups=groups,
                                   method="predict_proba")[:, 1]
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        probs = cross_val_predict(model, Xs, y, cv=cv,
                                   method="predict_proba")[:, 1]
    return probs


def evaluate_feature_set(X, y, groups, use_group_cv, n_splits, label):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    candidates = get_candidate_models()
    oof = {}
    accs = {}
    for name, model in candidates.items():
        probs = cv_probs(model, Xs, y, groups, use_group_cv, n_splits)
        oof[name] = probs
        accs[name] = accuracy_score(y, (probs >= 0.5).astype(int))
        print(f"    [{label}] {name:14s} accuracy: {accs[name]*100:.1f}%")

    names = list(oof.keys())
    if len(names) > 1:
        equal = np.mean([oof[n] for n in names], axis=0)
        oof["SoftVote(equal)"] = equal
        accs["SoftVote(equal)"] = accuracy_score(y, (equal >= 0.5).astype(int))
        print(f"    [{label}] {'SoftVote(equal)':14s} accuracy: "
              f"{accs['SoftVote(equal)']*100:.1f}%")

        w = np.array([accs[n] for n in names])
        w = w / w.sum()
        weighted = np.zeros_like(equal)
        for wi, n in zip(w, names):
            weighted += wi * oof[n]
        oof["WeightedAverage"] = weighted
        accs["WeightedAverage"] = accuracy_score(y, (weighted >= 0.5).astype(int))
        print(f"    [{label}] {'WeightedAverage':14s} accuracy: "
              f"{accs['WeightedAverage']*100:.1f}%  (weights: "
              + ", ".join(f'{n}={wi:.2f}' for n, wi in zip(names, w)) + ")")

    best_name = max(accs, key=accs.get)
    return {
        "scaler": scaler, "candidates": candidates,
        "oof_probs": oof, "accs": accs, "best_name": best_name,
        "best_acc": accs[best_name], "component_names": names,
    }


# --------------------------------------------------------------------------
# Threshold optimization
# --------------------------------------------------------------------------

def optimize_threshold(y, probs):
    thresholds = np.linspace(0.05, 0.95, 91)
    best = {"accuracy": (0.5, 0), "f1": (0.5, 0), "balanced_accuracy": (0.5, 0)}
    for t in thresholds:
        pred = (probs >= t).astype(int)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        bacc = balanced_accuracy_score(y, pred)
        if acc > best["accuracy"][1]:
            best["accuracy"] = (t, acc)
        if f1 > best["f1"][1]:
            best["f1"] = (t, f1)
        if bacc > best["balanced_accuracy"][1]:
            best["balanced_accuracy"] = (t, bacc)
    return best


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------

def error_analysis(y, probs, threshold, paths, out_dir):
    fp_dir = os.path.join(out_dir, "false_positives")
    fn_dir = os.path.join(out_dir, "false_negatives")
    for d in (fp_dir, fn_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    pred = (probs >= threshold).astype(int)
    fp_count, fn_count = 0, 0
    seen = set()
    for yi, pi, prob, path in zip(y, pred, probs, paths):
        key = (path, yi, pi)
        if key in seen:  # avoid dumping every augmented duplicate
            continue
        seen.add(key)
        if yi == 0 and pi == 1:  # real misclassified as screen
            dst = os.path.join(fp_dir, f"{prob:.2f}__{os.path.basename(path)}")
            shutil.copy(path, dst)
            fp_count += 1
        elif yi == 1 and pi == 0:  # screen misclassified as real
            dst = os.path.join(fn_dir, f"{prob:.2f}__{os.path.basename(path)}")
            shutil.copy(path, dst)
            fn_count += 1
    print(f"  Error analysis: {fp_count} unique false positives -> {fp_dir}")
    print(f"                  {fn_count} unique false negatives -> {fn_dir}")

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["real", "screen"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=120)
    plt.close(fig)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y, probs)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=120)
    plt.close(fig)

    print(f"  Plots saved: confusion_matrix.png, roc_curve.png, pr_curve.png -> {out_dir}")
    return roc_auc


def plot_feature_importance(names, importances, out_dir, top_n=20):
    order = np.argsort(-importances)[:top_n]
    fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.28)))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance (top {})".format(top_n))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=120)
    plt.close(fig)
    print(f"  Feature importance plot -> {out_dir}/feature_importance.png")


# --------------------------------------------------------------------------
# Calibration + final model assembly
# --------------------------------------------------------------------------

def calibrate_model(model, Xs, y, groups, use_group_cv, n_splits=3):
    """Wrap a fresh (unfitted) model in CalibratedClassifierCV so predicted
    probabilities are more reliable, using grouped CV internally so
    calibration itself doesn't leak augmented near-duplicates.

    ensemble=False: fits the base estimator ONCE on all the data and only
    cross-fits the (tiny) calibration curve, instead of storing one full
    base-estimator copy per CV fold. Keeps the saved model close to the
    size of a single classifier rather than n_splits times larger - matters
    a lot for the "<1MB, mobile-friendly" goal with a 300-tree forest."""
    if use_group_cv:
        cv_splits = list(GroupKFold(n_splits=n_splits).split(Xs, y, groups))
    else:
        cv_splits = n_splits
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_splits,
                                         ensemble=False)
    calibrated.fit(Xs, y)
    return calibrated


def build_final_model(result, X, y, groups, use_group_cv):
    """Fit the winning candidate (single model or ensemble) on ALL data,
    with each base learner calibrated."""
    best_name = result["best_name"]
    scaler = result["scaler"]
    Xs = scaler.transform(X)  # scaler already fit during evaluate_feature_set

    if best_name in ("SoftVote(equal)", "WeightedAverage"):
        component_names = result["component_names"]
        calibrated_models = []
        for name in component_names:
            fresh = get_candidate_models()[name]
            calibrated_models.append(calibrate_model(fresh, Xs, y, groups, use_group_cv))
        if best_name == "SoftVote(equal)":
            weights = None
        else:
            w = np.array([result["accs"][n] for n in component_names])
            weights = (w / w.sum()).tolist()
        final_model = EnsembleModel(calibrated_models, weights=weights, names=component_names)
    else:
        fresh = get_candidate_models()[best_name]
        final_model = calibrate_model(fresh, Xs, y, groups, use_group_cv)

    return final_model


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--augment", action="store_true",
                     help="apply quick in-memory augmentation (use with raw data/)")
    ap.add_argument("--group_cv", action="store_true",
                     help="force group-aware CV (auto-enabled if augmented data is detected)")
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--error_dir", default="error_analysis")
    ap.add_argument("--feature_cum_importance", type=float, default=0.90,
                     help="cumulative importance threshold for feature selection")
    args = ap.parse_args()

    print("Loading + extracting features...")
    t0 = time.time()
    X, y, groups, paths = build_dataset(args.data_dir, augment=args.augment)
    n_unique_sources = len(set(groups))
    print(f"Total samples: {len(y)}  (real={sum(y==0)}, screen={sum(y==1)})"
          f"  from {n_unique_sources} unique source photos"
          f"  [{time.time()-t0:.1f}s]")

    if len(set(y)) < 2 or len(y) < 10:
        print("Not enough data yet. Add photos to data/real/ and data/screen/ "
              "(aim for ~50 each) and re-run.")
        return

    use_group_cv = args.group_cv or (n_unique_sources < len(y))
    n_splits = max(2, min(5, min(np.bincount(y)), n_unique_sources))
    if use_group_cv:
        print(f"Using GroupKFold ({n_splits}-fold, grouped by source photo) "
              f"to avoid train/test leakage between augmented copies.")

    # ---- Feature importance + selection ----
    print("\nRanking feature importance (full feature set)...")
    scaler_full = StandardScaler()
    Xs_full = scaler_full.fit_transform(X)
    order, importances = rank_feature_importance(Xs_full, y)
    print("Top 10 features:")
    for i in order[:10]:
        print(f"  {FEATURE_NAMES[i]:26s} {importances[i]:.3f}")

    keep_idx = select_top_features(order, importances, args.feature_cum_importance)
    print(f"\nFeature selection: keeping {len(keep_idx)}/{len(FEATURE_NAMES)} features "
          f"(>= {args.feature_cum_importance*100:.0f}% cumulative importance)")

    # ---- Compare full vs reduced feature set ----
    print("\nEvaluating FULL feature set:")
    result_full = evaluate_feature_set(X, y, groups, use_group_cv, n_splits, "full")
    print("\nEvaluating REDUCED feature set:")
    X_reduced = X[:, keep_idx]
    result_reduced = evaluate_feature_set(X_reduced, y, groups, use_group_cv, n_splits, "reduced")

    # prefer the reduced set unless it's meaningfully worse (>0.5pp)
    if result_reduced["best_acc"] >= result_full["best_acc"] - 0.005:
        print(f"\n>> Using REDUCED feature set ({len(keep_idx)} features): "
              f"{result_reduced['best_acc']*100:.1f}% vs full "
              f"{result_full['best_acc']*100:.1f}%")
        chosen_X, result, feature_indices = X_reduced, result_reduced, keep_idx
    else:
        print(f"\n>> Using FULL feature set: reduced set cost too much accuracy "
              f"({result_reduced['best_acc']*100:.1f}% vs {result_full['best_acc']*100:.1f}%)")
        chosen_X, result, feature_indices = X, result_full, list(range(len(FEATURE_NAMES)))

    best_name = result["best_name"]
    best_probs = result["oof_probs"][best_name]
    print(f"\n=== Best approach: {best_name}  "
          f"({'group-' if use_group_cv else ''}CV accuracy: {result['best_acc']*100:.1f}%) ===")
    print(classification_report(y, (best_probs >= 0.5).astype(int),
                                 target_names=["real", "screen"]))

    # ---- Threshold optimization ----
    thresholds = optimize_threshold(y, best_probs)
    print("Threshold optimization (on out-of-fold probabilities):")
    for metric, (t, val) in thresholds.items():
        print(f"  best {metric:18s}: threshold={t:.2f}  {metric}={val*100:.1f}%")
    chosen_threshold = thresholds["balanced_accuracy"][0]
    print(f"  -> using balanced-accuracy-optimal threshold: {chosen_threshold:.2f} "
          f"(default was 0.5)")

    # ---- Error analysis ----
    os.makedirs(args.error_dir, exist_ok=True)
    roc_auc = error_analysis(y, best_probs, chosen_threshold, paths, args.error_dir)
    print(f"  ROC AUC: {roc_auc:.3f}")

    selected_names = [FEATURE_NAMES[i] for i in feature_indices]
    if hasattr(result["candidates"].get(best_name), "feature_importances_") or \
       best_name in result["candidates"]:
        # importance for plotting: use the single-model importances if
        # available, else fall back to full-set RF importances restricted
        # to the selected features
        plot_feature_importance(selected_names,
                                 importances[feature_indices], args.error_dir)

    # ---- Calibrate + fit final model on ALL data ----
    print("\nCalibrating probabilities and fitting final model on all data...")
    final_model = build_final_model(result, chosen_X, y, groups, use_group_cv)

    joblib.dump({
        "scaler": result["scaler"],
        "model": final_model,
        "model_name": best_name,
        "feature_indices": feature_indices,
        "cv_accuracy": result["best_acc"],
        "roc_auc": roc_auc,
        "thresholds": thresholds,
        "operating_threshold": chosen_threshold,
        "group_cv": use_group_cv,
        "calibrated": True,
    }, args.out)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nSaved model ({best_name}, calibrated) to {args.out}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()