"""
augment.py
----------
Offline augmentation: reads your raw photos from data/real/ and
data/screen/ and writes an EXPANDED set of augmented images to disk under
data_augmented/real/ and data_augmented/screen/ (roughly 500-600 total from
~100 originals).

Why offline (saved to disk) instead of the in-memory version:
- you can visually spot-check the augmented images before training
- train.py's cross-validation can group by source photo (see --group_cv)
  to avoid leaking near-duplicate augmented copies across train/test folds,
  which would otherwise make the reported accuracy dishonest

Augmentations used (each mild, so the recapture "signal" in the features
survives): rotation, horizontal flip, brightness/contrast jitter, gamma
adjustment, slight blur, and JPEG re-compression at a random quality.

Usage:
    python augment.py --data_dir data --out_dir data_augmented --target_total 550
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def load_paths(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def brightness_contrast(img, alpha, beta):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gamma_adjust(img, gamma):
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def slight_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def jpeg_recompress(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def small_perspective(img, jitter=0.03):
    """Mild perspective warp - simulates a slightly non-frontal capture
    angle, which is common and realistic for both real and screen photos."""
    h, w = img.shape[:2]
    d = jitter * min(h, w)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = src + np.random.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def mild_affine(img, shear=0.04):
    """Small shear/skew - simulates hand-held camera tilt."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h]])
    dst = np.float32([
        [0 + shear * w * np.random.uniform(-1, 1), 0],
        [w, 0 + shear * h * np.random.uniform(-1, 1)],
        [0, h],
    ])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def scale_change(img, factor):
    """Slight zoom in/out then crop/pad back to original size - simulates
    photographing from a different distance."""
    h, w = img.shape[:2]
    nh, nw = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (nw, nh))
    if factor >= 1.0:
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        return resized[y0:y0 + h, x0:x0 + w]
    else:
        pad_h, pad_w = h - nh, w - nw
        top, left = pad_h // 2, pad_w // 2
        return cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                   cv2.BORDER_REFLECT)


def exposure_variation(img, ev):
    """Slight exposure shift in linear-ish light (gentler than plain
    brightness offset - multiplies rather than adds)."""
    factor = 2.0 ** ev  # ev in stops, e.g. +-0.3
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# Each recipe is a short chain of 1-2 mild transforms. Kept deliberately
# varied (not every combination) so augmented copies still look like
# plausible different real-world captures rather than degenerate noise.
RECIPES = [
    lambda im: im,  # original, unmodified copy (kept as a "0" augmentation)
    lambda im: cv2.flip(im, 1),
    lambda im: rotate(im, 6),
    lambda im: rotate(im, -6),
    lambda im: rotate(im, 12),
    lambda im: rotate(im, -12),
    lambda im: brightness_contrast(im, 1.2, 12),
    lambda im: brightness_contrast(im, 0.8, -12),
    lambda im: brightness_contrast(im, 1.35, 0),
    lambda im: gamma_adjust(im, 1.4),
    lambda im: gamma_adjust(im, 0.7),
    lambda im: slight_blur(im, 3),
    lambda im: jpeg_recompress(im, 40),
    lambda im: jpeg_recompress(im, 25),
    lambda im: cv2.flip(rotate(im, 8), 1),
    lambda im: jpeg_recompress(brightness_contrast(im, 1.15, -8), 50),
    lambda im: gamma_adjust(cv2.flip(im, 1), 1.25),
    lambda im: slight_blur(rotate(im, -9), 3),
    lambda im: small_perspective(im, 0.03),
    lambda im: small_perspective(im, 0.05),
    lambda im: mild_affine(im, 0.04),
    lambda im: mild_affine(im, 0.06),
    lambda im: scale_change(im, 1.1),
    lambda im: scale_change(im, 0.9),
    lambda im: exposure_variation(im, 0.3),
    lambda im: exposure_variation(im, -0.3),
    lambda im: small_perspective(rotate(im, 5), 0.02),
    lambda im: exposure_variation(scale_change(im, 1.05), -0.2),
]


def augment_one(img, n_variants, rng):
    """Return n_variants augmented copies of img using random recipes
    (without replacement where possible)."""
    recipes = RECIPES.copy()
    rng.shuffle(recipes)
    out = []
    i = 0
    while len(out) < n_variants:
        recipe = recipes[i % len(recipes)]
        try:
            out.append(recipe(img))
        except Exception:
            pass
        i += 1
        if i > n_variants * 3:  # safety valve
            break
    return out[:n_variants]


def process_folder(src_folder, dst_folder, target_total, rng):
    os.makedirs(dst_folder, exist_ok=True)
    paths = load_paths(src_folder)
    if not paths:
        print(f"  [warn] no images found in {src_folder}")
        return

    n_variants = max(1, round(target_total / len(paths)))
    print(f"  {src_folder}: {len(paths)} originals -> "
          f"~{n_variants} variants each (~{len(paths)*n_variants} total)")

    count = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [skip, unreadable] {p}")
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        variants = augment_one(img, n_variants, rng)
        for vi, v in enumerate(variants):
            out_name = f"{base}__aug{vi:02d}.jpg"
            cv2.imwrite(os.path.join(dst_folder, out_name), v,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            count += 1
    print(f"  wrote {count} images to {dst_folder}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="data_augmented")
    ap.add_argument("--target_total", type=int, default=550,
                     help="approx total images PER CLASS after augmentation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Generating offline-augmented dataset...")
    process_folder(os.path.join(args.data_dir, "real"),
                    os.path.join(args.out_dir, "real"),
                    args.target_total, rng)
    process_folder(os.path.join(args.data_dir, "screen"),
                    os.path.join(args.out_dir, "screen"),
                    args.target_total, rng)
    print(f"\nDone. Train with:\n"
          f"  python train.py --data_dir {args.out_dir} --group_cv")


if __name__ == "__main__":
    main()