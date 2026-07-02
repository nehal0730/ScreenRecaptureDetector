"""
extract_features.py
--------------------
Turns one image into a small vector of numbers that capture the
tell-tale signs of a "photo of a screen" (recapture) vs a real photo:

1. Moire / periodicity energy   - screens have a regular pixel/subpixel
                                   grid; photographing it beats against
                                   the camera's own sensor grid and
                                   creates periodic patterns visible as
                                   sharp, off-center peaks in the 2D
                                   FFT. Real-world textures don't do this.
2. High-frequency energy ratio  - screens (esp. re-encoded/backlit) tend
                                   to have different high-freq energy
                                   distribution than natural scenes.
3. Sharpness (Laplacian var)    - recaptures are very often slightly
                                   softer/blurrier (extra glass + optics
                                   + refocus) than a direct real photo.
4. Texture regularity (LBP)     - Local Binary Pattern histogram entropy
                                   is lower/more regular for a pixel-grid
                                   surface than for natural material.
5. Color / white-balance shift  - screens (LCD/OLED/print) often skew
                                   blue or have compressed color gamut;
                                   we look at channel means, saturation,
                                   and blue-red balance.
6. Glare / specular highlights  - screens under normal room light very
                                   often show a bright specular blob or
                                   washed-out corner; real scenes rarely
                                   have such a hard-edged bright patch.
7. Noise-residual statistics    - subtract a blurred version of the image
                                   from itself; the leftover "noise" has
                                   different statistics for sensor noise
                                   (real) vs re-sampled/recaptured noise.
8. GLCM texture (contrast,        - Gray-Level Co-occurrence Matrix stats
   homogeneity, energy,             capture second-order texture regularity
   correlation)                     more precisely than LBP alone; screens'
                                     repeating grid tends to show up as
                                     unusually high homogeneity/energy.
9. Gradient statistics /          - mean/std of Sobel gradient magnitude and
   orientation entropy              the entropy of gradient-orientation
                                     histogram; recaptures often have a
                                     narrower, more axis-aligned gradient
                                     distribution (screen pixel grid) than
                                     natural scenes.
10. Color temperature (CCT)       - approximate correlated color temperature
                                     (McCamy's formula from RGB chromaticity);
                                     screens/printouts often run several
                                     hundred K cooler or warmer than the
                                     ambient-lit real scene around them.
11. Multi-scale FFT                - the same moire/periodicity stats (9)
                                      computed again at half- and quarter-
                                      resolution, since a screen's pixel
                                      grid can alias into different
                                      frequency bands depending on the
                                      distance/zoom the recapture was
                                      taken at. Cheap because the smaller
                                      scales are, well, smaller.
12. JPEG blockiness                - recaptures are almost always
                                      double-compressed (once by the
                                      original screen content, again by
                                      the camera); we measure how much
                                      stronger the gradient is exactly at
                                      8x8 JPEG block boundaries versus
                                      elsewhere, a classic compression-
                                      artifact tell.
13. Intensity/residual skew         - skewness and kurtosis of the pixel-
    & kurtosis                        intensity histogram and of the
                                       noise-residual histogram; screens
                                       tend to produce more sharply-peaked
                                       (leptokurtic) residual distributions
                                       than natural sensor noise.
14. Local contrast statistics       - mean/std of a local-variance map
                                       (blockwise contrast); backlit
                                       screens and glare tend to flatten
                                       local contrast unevenly compared to
                                       naturally lit scenes.

None of these features alone is reliable — that's the point of the
brief ("subtle clues"). Combined via a classifier they get us to a
useful accuracy while staying tiny and fast.
"""

import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

TARGET_SIZE = 256  # resize (with padding) so FFT bins are comparable across images
                    # (256 chosen for speed; still plenty of resolution for
                    # frequency/texture statistics - see note.md for the tradeoff)

GLCM_LEVELS = 32   # quantize grayscale to this many levels before GLCM
                    # (keeps the co-occurrence matrix small -> fast)

FFT_SCALES = (1.0, 0.5, 0.25)  # multi-scale FFT: full / half / quarter res

FEATURE_NAMES = [
    "fft_high_freq_ratio",
    "fft_peak_strength",
    "fft_peak_count",
    "laplacian_var",
    "edge_density",
    "lbp_entropy",
    "lbp_uniform_ratio",
    "mean_R", "mean_G", "mean_B",
    "std_R", "std_G", "std_B",
    "blue_red_diff",
    "sat_mean", "sat_std",
    "glare_ratio",
    "noise_residual_std",
    "noise_residual_mean_abs",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_asm",
    "grad_mag_mean",
    "grad_mag_std",
    "grad_orientation_entropy",
    "color_temp_k",
    # multi-scale FFT (half + quarter res; full-res already above)
    "fft_high_freq_ratio_half",
    "fft_peak_strength_half",
    "fft_peak_count_half",
    "fft_high_freq_ratio_quarter",
    "fft_peak_strength_quarter",
    "fft_peak_count_quarter",
    # JPEG blockiness
    "jpeg_blockiness",
    # distribution shape
    "intensity_skew",
    "intensity_kurtosis",
    "residual_skew",
    "residual_kurtosis",
    # local contrast
    "local_contrast_mean",
    "local_contrast_std",
]


def _resize_pad(img, size=TARGET_SIZE):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_REFLECT)
    return padded


def _fft_features(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # energy outside a low-frequency disk vs total energy
    low_mask = dist < (0.06 * min(h, w))
    total_energy = magnitude.sum() + 1e-8
    high_freq_ratio = magnitude[~low_mask].sum() / total_energy

    # look for sharp, isolated peaks away from the DC center
    # (periodic screen-grid / moire signature)
    ring_mask = (dist > 0.08 * min(h, w)) & (dist < 0.45 * min(h, w))
    ring_vals = magnitude[ring_mask]
    if ring_vals.size > 0:
        thresh = ring_vals.mean() + 3 * ring_vals.std()
        peak_count = int((ring_vals > thresh).sum())
        peak_strength = float(ring_vals.max() - ring_vals.mean())
    else:
        peak_count = 0
        peak_strength = 0.0

    return float(high_freq_ratio), peak_strength, peak_count


def _lbp_features(gray):
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    hist = hist + 1e-8
    entropy = float(-(hist * np.log(hist)).sum())
    uniform_ratio = float(hist[:-1].sum())  # uniform patterns vs "noise" bin
    return entropy, uniform_ratio


def _noise_residual(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(residual.std()), float(np.abs(residual).mean())


def _glcm_features(gray):
    # quantize to fewer gray levels so the co-occurrence matrix stays small/fast
    q = (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    q = np.clip(q, 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(
        q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS, symmetric=True, normed=True,
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())
    asm = float(graycoprops(glcm, "ASM").mean())
    return contrast, homogeneity, energy, correlation, asm


def _gradient_features(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_mean, mag_std = float(mag.mean()), float(mag.std())

    # orientation histogram entropy - a strong pixel-grid/moire pattern
    # concentrates gradient energy into a few axis-aligned bins, lowering
    # entropy relative to natural-scene gradients
    orient = np.arctan2(gy, gx)
    weights = mag.ravel()
    hist, _ = np.histogram(orient.ravel(), bins=9, range=(-np.pi, np.pi),
                            weights=weights, density=True)
    hist = hist + 1e-8
    hist = hist / hist.sum()
    orient_entropy = float(-(hist * np.log(hist)).sum())

    return mag_mean, mag_std, orient_entropy


def _color_temperature(img_bgr):
    # rough correlated color temperature (CCT) via McCamy's approximation
    # from mean-RGB -> CIE xy chromaticity. Good enough as a discriminative
    # feature, not meant to be colorimetrically exact.
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    R, G, B = r.mean() / 255.0, g.mean() / 255.0, b.mean() / 255.0

    # simple linearization + sRGB->XYZ
    def lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    Rl, Gl, Bl = lin(R), lin(G), lin(B)
    X = 0.4124 * Rl + 0.3576 * Gl + 0.1805 * Bl
    Y = 0.2126 * Rl + 0.7152 * Gl + 0.0722 * Bl
    Z = 0.0193 * Rl + 0.1192 * Gl + 0.9505 * Bl

    denom = (X + Y + Z) + 1e-8
    x = X / denom
    y = Y / denom

    denom2 = (0.1858 - y)
    denom2 = denom2 if abs(denom2) > 1e-6 else 1e-6
    n = (x - 0.3320) / denom2
    cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    cct = float(np.clip(cct, 1000, 20000))  # clamp to a sane range
    return cct


def _multiscale_fft_features(gray):
    """FFT moire/periodicity stats at half and quarter resolution (full-res
    is already computed separately). Cheap since the arrays shrink fast."""
    out = []
    h, w = gray.shape
    for scale in FFT_SCALES[1:]:  # skip 1.0, already computed at full res
        small = cv2.resize(gray, (max(8, int(w * scale)), max(8, int(h * scale))),
                            interpolation=cv2.INTER_AREA)
        hf_ratio, peak_strength, peak_count = _fft_features(small)
        out.extend([hf_ratio, peak_strength, peak_count])
    return out


def _jpeg_blockiness(gray):
    """Ratio of gradient energy at 8x8 JPEG block boundaries vs elsewhere.
    Re-encoded (recaptured) images tend to show stronger periodic blocking."""
    gray_f = gray.astype(np.float32)
    gx = np.abs(np.diff(gray_f, axis=1))
    gy = np.abs(np.diff(gray_f, axis=0))

    h, w = gray_f.shape
    col_idx = np.arange(gx.shape[1])
    row_idx = np.arange(gy.shape[0])

    boundary_cols = (col_idx % 8 == 7)
    boundary_rows = (row_idx % 8 == 7)

    boundary_energy = gx[:, boundary_cols].mean() + gy[boundary_rows, :].mean()
    other_energy = gx[:, ~boundary_cols].mean() + gy[~boundary_rows, :].mean()

    return float(boundary_energy / (other_energy + 1e-6))


def _distribution_shape(gray, residual):
    intensity_skew = float(skew(gray.ravel().astype(np.float64)))
    intensity_kurt = float(kurtosis(gray.ravel().astype(np.float64)))
    residual_skew = float(skew(residual.ravel()))
    residual_kurt = float(kurtosis(residual.ravel()))
    return intensity_skew, intensity_kurt, residual_skew, residual_kurt


def _local_contrast(gray, block=16):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (block, block))
    sqmean = cv2.blur(gray_f * gray_f, (block, block))
    local_var = np.maximum(sqmean - mean * mean, 0)
    local_std = np.sqrt(local_var)
    return float(local_std.mean()), float(local_std.std())


def extract_features(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_array}")
    else:
        img = image_path_or_array

    img = _resize_pad(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    high_freq_ratio, peak_strength, peak_count = _fft_features(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 80, 160)
    edge_density = edges.mean() / 255.0

    lbp_entropy, lbp_uniform_ratio = _lbp_features(gray)

    b, g, r = cv2.split(img.astype(np.float32))
    mean_R, mean_G, mean_B = r.mean(), g.mean(), b.mean()
    std_R, std_G, std_B = r.std(), g.std(), b.std()
    blue_red_diff = mean_B - mean_R

    sat = hsv[:, :, 1].astype(np.float32)
    sat_mean, sat_std = sat.mean(), sat.std()

    val = hsv[:, :, 2]
    glare_ratio = float((val > 250).mean())

    noise_std, noise_mean_abs = _noise_residual(gray)

    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm = \
        _glcm_features(gray)

    grad_mag_mean, grad_mag_std, grad_orient_entropy = _gradient_features(gray)

    color_temp_k = _color_temperature(img)

    (fft_hf_half, fft_peak_half, fft_count_half,
     fft_hf_quarter, fft_peak_quarter, fft_count_quarter) = \
        _multiscale_fft_features(gray)

    jpeg_blockiness = _jpeg_blockiness(gray)

    blurred_for_residual = cv2.GaussianBlur(gray, (5, 5), 0)
    residual_map = gray.astype(np.float32) - blurred_for_residual.astype(np.float32)
    intensity_skew, intensity_kurt, residual_skew, residual_kurt = \
        _distribution_shape(gray, residual_map)

    local_contrast_mean, local_contrast_std = _local_contrast(gray)

    feats = np.array([
        high_freq_ratio, peak_strength, peak_count,
        laplacian_var, edge_density,
        lbp_entropy, lbp_uniform_ratio,
        mean_R, mean_G, mean_B,
        std_R, std_G, std_B,
        blue_red_diff,
        sat_mean, sat_std,
        glare_ratio,
        noise_std, noise_mean_abs,
        glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm,
        grad_mag_mean, grad_mag_std, grad_orient_entropy,
        color_temp_k,
        fft_hf_half, fft_peak_half, fft_count_half,
        fft_hf_quarter, fft_peak_quarter, fft_count_quarter,
        jpeg_blockiness,
        intensity_skew, intensity_kurt, residual_skew, residual_kurt,
        local_contrast_mean, local_contrast_std,
    ], dtype=np.float64)

    return feats