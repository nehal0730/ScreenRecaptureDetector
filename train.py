"""
train.py
--------
Trains the full inference pipeline used by predict.py / app.py:

    Preprocessing -> Feature Extraction -> Feature Selection ->
    Model Selection -> Probability Calibration -> Threshold Optimization
    -> Prediction

  STAGE 1 - Preprocessing (preprocessing.py): load, validate (reject
      corrupted/too-small/blank files), normalize color format (grayscale
      / RGBA -> BGR), resize+letterbox to a fixed resolution regardless of
      portrait/landscape orientation.
  STAGE 1b - Feature Extraction (extract_features.py): 41 handcrafted
      features computed on the clean, preprocessed image.
  STAGE 2 - Feature Selection: rank features by RandomForest importance,
      then search for the SMALLEST feature subset whose cross-validated
      accuracy is within a small tolerance of the best accuracy any subset
      size achieves - not a fixed importance percentage.
  STAGE 3 - Model Selection: compare RandomForest, XGBoost, LightGBM (if
      installed), plus soft-voting and accuracy-weighted ensembles of
      them, via GroupKFold (grouped by source photo, so augmented copies
      never leak across train/test) - keep whichever wins.
  STAGE 4 - Probability Calibration: wrap the winner in
      CalibratedClassifierCV (Platt/sigmoid scaling).
  STAGE 5 - Threshold Optimization: search for the decision threshold
      that maximizes balanced accuracy (also reports accuracy/F1-optimal
      thresholds) instead of assuming 0.5.
  (STAGE 6 - Prediction happens in predict.py / app.py at inference time,
      using everything saved to model.pkl by this script.)

Also writes error-analysis artifacts (confusion matrix, ROC, PR curve,
probability distribution, feature importance plot, misclassified images)
so failure modes are visible, not just a single accuracy number.

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
import preprocessing

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
    n_skipped = 0
    for label, folder_name in ((0, "real"), (1, "screen")):
        folder = os.path.join(data_dir, folder_name)
        paths = load_paths(folder)
        print(f"  {folder_name}: {len(paths)} images")
        for p in paths:
            # Stage 1 (Preprocessing) robustness check: reject corrupted,
            # unreadable, degenerate, or too-small files before they ever
            # reach feature extraction, instead of letting a bad file
            # crash the whole training run.
            try:
                img = preprocessing.load_image(p)
                img = preprocessing.normalize_color(img)
                img = preprocessing.validate_content(img, source_label=p)
            except preprocessing.ImageValidationError as e:
                print(f"  [skip, invalid] {e}")
                n_skipped += 1
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
    if n_skipped:
        print(f"  Skipped {n_skipped} invalid/corrupted image(s) total.")
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


def select_smallest_stable_subset(X, y, groups, use_group_cv, n_splits, order,
                                   tolerance=0.005, min_features=3):
    """Instead of a fixed cumulative-importance cutoff, actually search for
    the smallest feature count whose cross-validated accuracy is within
    `tolerance` of the best accuracy achieved by any prefix of the
    importance-ranked feature list. Uses a fast RandomForest as the search
    proxy (the real model/ensemble comparison happens afterward, only on
    the winning subset) - cheap enough to evaluate every feature count on
    a dataset this size, so no need to approximate with a fixed threshold."""
    proxy = RandomForestClassifier(
        n_estimators=150, max_depth=7, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    ks = list(range(min_features, len(order) + 1))
    accs = []
    for k in ks:
        idx = sorted(order[:k].tolist())
        scaler = StandardScaler()
        Xk_s = scaler.fit_transform(X[:, idx])
        probs = cv_probs(proxy, Xk_s, y, groups, use_group_cv, n_splits)
        accs.append(accuracy_score(y, (probs >= 0.5).astype(int)))
    accs = np.array(accs)
    best_acc = accs.max()
    stable = accs >= (best_acc - tolerance)
    chosen_pos = int(np.argmax(stable))  # first k within tolerance of the peak
    chosen_k = ks[chosen_pos]
    keep_idx = sorted(order[:chosen_k].tolist())
    curve = list(zip(ks, accs.tolist()))
    return keep_idx, curve, best_acc


# --------------------------------------------------------------------------
# Classifier + ensemble comparison
# --------------------------------------------------------------------------

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

def get_candidate_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        print("  [info] xgboost not installed - skipping "
              "(pip install xgboost to include it)")
    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=1200,
            depth=8,
            learning_rate=0.02,
            l2_leaf_reg=3,
            random_strength=1,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=42
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
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


def plot_probability_distribution(y, probs, threshold, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(probs[y == 0], bins=bins, alpha=0.6, label="real", color="#1F8A70")
    ax.hist(probs[y == 1], bins=bins, alpha=0.6, label="screen", color="#A93226")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"threshold={threshold:.2f}")
    ax.set_xlabel("P(screen)")
    ax.set_ylabel("count")
    ax.set_title("Predicted probability distribution by true class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probability_distribution.png"), dpi=120)
    plt.close(fig)
    print(f"  Probability distribution plot -> {out_dir}/probability_distribution.png")


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
    ap = argparse.ArgumentParser(description=(
        "Inference pipeline this trains: Preprocessing -> Feature Extraction "
        "-> Feature Selection -> Model Selection -> Probability Calibration "
        "-> Threshold Optimization -> Prediction. See note.md for details."
    ))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--augment", action="store_true",
                     help="apply quick in-memory augmentation (use with raw data/)")
    ap.add_argument("--group_cv", action="store_true",
                     help="force group-aware CV (auto-enabled if augmented data is detected)")
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--error_dir", default="error_analysis")
    ap.add_argument("--selection_tolerance", type=float, default=0.005,
                     help="keep the smallest feature subset within this much "
                          "accuracy of the best subset found (default 0.5pp)")
    args = ap.parse_args()

    print("=" * 70)
    print("STAGE 1: Preprocessing + Feature Extraction")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("STAGE 2: Feature Selection")
    print("=" * 70)
    print("Ranking feature importance (full feature set)...")
    scaler_full = StandardScaler()
    Xs_full = scaler_full.fit_transform(X)
    order, importances = rank_feature_importance(Xs_full, y)
    print("Top 10 features:")
    for i in order[:10]:
        print(f"  {FEATURE_NAMES[i]:26s} {importances[i]:.3f}")

    print(f"\nSearching for the smallest feature subset within "
          f"{args.selection_tolerance*100:.1f}pp of peak CV accuracy "
          f"(not a fixed importance percentage)...")
    feature_indices, curve, peak_acc = select_smallest_stable_subset(
        X, y, groups, use_group_cv, n_splits, order,
        tolerance=args.selection_tolerance)
        
    # print a compact view of the accuracy-vs-feature-count curve
    for k, acc in curve:
        if k == feature_indices.__len__() or k % 5 == 0 or k == curve[-1][0]:
            marker = "  <- chosen" if k == len(feature_indices) else ""
            print(f"  {k:2d} features: {acc*100:5.1f}%{marker}")
    print(f">> Selected {len(feature_indices)}/{len(FEATURE_NAMES)} features "
          f"(peak CV accuracy across all subset sizes was {peak_acc*100:.1f}%)")

    chosen_X = X[:, feature_indices]
    # print("\nUsing ALL handcrafted features (feature selection disabled).")
    # chosen_X = X
    # feature_indices = list(range(X.shape[1]))

    print("\n" + "=" * 70)
    print("STAGE 3: Model Selection")
    print("=" * 70)
    result = evaluate_feature_set(chosen_X, y, groups, use_group_cv, n_splits, "selected")

    best_name = result["best_name"]
    best_probs = result["oof_probs"][best_name]
    print(f"\n=== Best approach: {best_name}  "
          f"({'group-' if use_group_cv else ''}CV accuracy: {result['best_acc']*100:.1f}%) ===")
    print(classification_report(y, (best_probs >= 0.5).astype(int),
                                 target_names=["real", "screen"]))

    print("\n" + "=" * 70)
    print("STAGE 4/5: Probability Calibration + Threshold Optimization")
    print("=" * 70)
    thresholds = optimize_threshold(y, best_probs)
    print("Threshold search (on out-of-fold probabilities):")
    for metric, (t, val) in thresholds.items():
        print(f"  best {metric:18s}: threshold={t:.2f}  {metric}={val*100:.1f}%")
    chosen_threshold = thresholds["balanced_accuracy"][0]
    print(f"  -> using balanced-accuracy-optimal threshold: {chosen_threshold:.2f} "
          f"(default was 0.5)")

    print("\n" + "=" * 70)
    print("Error analysis")
    print("=" * 70)
    os.makedirs(args.error_dir, exist_ok=True)
    roc_auc = error_analysis(y, best_probs, chosen_threshold, paths, args.error_dir)
    print(f"  ROC AUC: {roc_auc:.3f}")
    plot_probability_distribution(y, best_probs, chosen_threshold, args.error_dir)

    selected_names = [FEATURE_NAMES[i] for i in feature_indices]
    plot_feature_importance(selected_names, importances[feature_indices], args.error_dir)

    print("\nCalibrating probabilities (Platt/sigmoid) and fitting final model "
          "on all data...")
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
        "pipeline": "Preprocessing -> Feature Extraction -> Feature Selection "
                    "-> Model Selection -> Probability Calibration -> "
                    "Threshold Optimization -> Prediction",
    }, args.out)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nSaved model ({best_name}, calibrated) to {args.out}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()