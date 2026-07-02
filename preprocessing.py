"""
preprocessing.py
-----------------
Stage 1 of the pipeline: Preprocessing -> Feature Extraction -> Feature
Selection -> Model Selection -> Probability Calibration -> Threshold
Optimization -> Prediction  (see note.md for the full pipeline diagram).

Everything that touches a raw image before any feature is computed lives
here, so extract_features.py can assume a clean, validated, consistently
formatted BGR array. Concretely this stage:

  1. Loads the image (from a file path OR an already-decoded array, e.g.
     a frame captured by the live camera demo) and validates it isn't
     corrupted/empty/unreadable.
  2. Normalizes color format: grayscale -> BGR, BGRA/RGBA (PNG with alpha)
     -> BGR, so every downstream feature always sees a 3-channel image.
  3. Resizes to a fixed working resolution while preserving aspect ratio
     (letterbox/pad rather than stretch), which also makes portrait and
     landscape captures behave identically - no separate orientation
     handling needed once this step is done.

Keeping this as its own module (rather than inline in extract_features.py)
means train.py's batch loader and app.py's live-frame handler both get the
exact same robustness guarantees for free.
"""

import os

import cv2
import numpy as np

TARGET_SIZE = 256  # fixed working resolution after preprocessing (see
                    # note.md for the speed/accuracy tradeoff discussion)

MIN_DIMENSION = 16  # images smaller than this in either axis are rejected
                     # as too degenerate to extract meaningful features from


class ImageValidationError(ValueError):
    """Raised when an input image is unreadable, corrupted, or degenerate."""
    pass


def load_image(image_path_or_array):
    """Load from a file path or accept an already-decoded BGR array (as
    produced by cv2.imdecode in app.py for live camera frames). Raises
    ImageValidationError with a clear message on any failure - callers
    (train.py's batch loader, predict.py, app.py) can catch this to skip
    a bad file gracefully instead of crashing the whole run."""
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            raise ImageValidationError(f"file not found: {image_path_or_array}")
        img = cv2.imread(image_path_or_array, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ImageValidationError(
                f"could not decode image (corrupted or unsupported format): "
                f"{image_path_or_array}")
    elif isinstance(image_path_or_array, np.ndarray):
        img = image_path_or_array
    else:
        raise ImageValidationError(
            f"unsupported input type: {type(image_path_or_array)}")
    return img


def normalize_color(img):
    """Handle grayscale, RGBA/BGRA (alpha-channel PNGs), and standard color
    images uniformly -> always returns a 3-channel BGR uint8 array."""
    if img.ndim == 2:
        # plain grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if channels == 3:
            return img
    raise ImageValidationError(f"unsupported image shape: {img.shape}")


def validate_content(img, source_label=""):
    """Reject degenerate images (too small, empty/blank/near-uniform,
    NaN/inf from a bad decode) before they reach feature extraction."""
    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageValidationError(
            f"image too small ({w}x{h}), minimum is {MIN_DIMENSION}px: {source_label}")
    if not np.isfinite(img).all():
        raise ImageValidationError(f"image contains NaN/Inf pixel values: {source_label}")
    if img.dtype != np.uint8:
        # normalize any other dtype (e.g. 16-bit PNG) into uint8 range
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if float(np.std(img)) < 1e-3:
        raise ImageValidationError(
            f"image is blank/near-uniform (std={np.std(img):.4f}), likely a bad capture: {source_label}")
    return img


def resize_and_normalize(img, target_size=TARGET_SIZE):
    """Letterbox-resize to a fixed square resolution, preserving aspect
    ratio (pads with reflected border rather than stretching) - this is
    what makes portrait and landscape captures behave identically without
    any separate orientation-specific logic."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (target_size - nh) // 2
    bottom = target_size - nh - top
    left = (target_size - nw) // 2
    right = target_size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_REFLECT)
    return padded


def preprocess_image(image_path_or_array, target_size=TARGET_SIZE):
    """The full preprocessing stage: load -> validate -> normalize color
    -> validate content -> resize. Returns a clean (target_size,
    target_size, 3) uint8 BGR array ready for feature extraction.

    Raises ImageValidationError on any problem, with a message specific
    enough to act on (corrupted file, too small, blank, bad format)."""
    label = image_path_or_array if isinstance(image_path_or_array, str) else "<array input>"
    img = load_image(image_path_or_array)
    img = normalize_color(img)
    img = validate_content(img, source_label=label)
    img = resize_and_normalize(img, target_size)
    return img


def is_valid_image(image_path_or_array):
    """Quick boolean check (no exception) - handy for a fast pre-filter
    over a large batch of files before the real (slower) processing."""
    try:
        preprocess_image(image_path_or_array)
        return True
    except ImageValidationError:
        return False