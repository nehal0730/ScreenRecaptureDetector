# Spot the Fake Photo — Notes

## Approach
No deep model, no GPU, by default. 41 hand-crafted features feed into a
pipeline that automatically: ranks + prunes features, compares
RandomForest/XGBoost/LightGBM plus soft-voting/weighted-average ensembles,
picks the best by honest group-aware cross-validation, optimizes the
decision threshold instead of assuming 0.5, calibrates the output
probabilities, and dumps error-analysis plots + misclassified images.

### Features (41 total)
- **Moire/periodicity (FFT)**, computed at **3 scales** (full/half/quarter
  resolution) — a screen's pixel grid beats against the camera sensor grid
  differently depending on capture distance/zoom, so checking multiple
  scales catches more cases than one fixed resolution.
- **Texture**: LBP entropy + GLCM (contrast, homogeneity, energy,
  correlation, ASM) — screen surfaces have more regular micro-texture.
- **Gradient statistics** — Sobel magnitude mean/std + orientation-entropy.
- **JPEG blockiness** — recaptures are almost always double-compressed;
  measures how much stronger the gradient is exactly at 8×8 JPEG block
  boundaries vs. elsewhere, a classic compression-artifact tell.
- **Distribution shape** — skewness/kurtosis of both the pixel-intensity
  histogram and the noise-residual histogram.
- **Local contrast** — mean/std of a blockwise local-variance map (screens
  and glare tend to flatten contrast unevenly vs. natural lighting).
- **Sharpness** (Laplacian variance), **color/color-temperature** (channel
  stats + McCamy CCT approximation), **glare ratio**, **noise-residual**
  stats — unchanged from the original set.

I did *not* add a separate full edge-orientation-histogram feature block
(HOG-style) on top of `grad_orientation_entropy` — the entropy already
compactly captures "how concentrated the gradient directions are," and a
full multi-bin histogram would mostly add redundant, correlated columns
for a 41-feature/~500-sample regime. The automatic feature-selection step
below would likely prune most of those bins anyway.

## Feature selection
`train.py` fits a RandomForest on the full feature set, ranks features by
impurity-based importance, and finds the smallest subset covering 90% of
cumulative importance. It then runs the **entire** classifier/ensemble
comparison on both the full and reduced feature sets and keeps the reduced
set unless it costs more than 0.5 percentage points of accuracy — smaller
feature vectors mean less to compute on a phone. The full importance
ranking is printed and also saved as `error_analysis/feature_importance.png`.

## Classifier + ensemble comparison
For each feature set, `train.py` cross-validates RandomForest and (if
installed) XGBoost/LightGBM individually, then also evaluates **soft
voting** (equal-weight average of their out-of-fold probabilities) and
**accuracy-weighted averaging**, and keeps whichever candidate — single
model or ensemble — scores highest. In my test environment only
RandomForest was available (no network access to install the extras
where I built this — see the honesty note below); the mechanism is fully
wired up and will include XGBoost/LightGBM and the ensemble comparisons
automatically once you `pip install xgboost lightgbm`.

## Threshold optimization
Rather than assuming 0.5, `train.py` scans thresholds from 0.05 to 0.95 on
out-of-fold probabilities and reports the best threshold for accuracy, F1,
and balanced accuracy separately. It ships the balanced-accuracy-optimal
threshold as the default operating point (`operating_threshold` in
`model.pkl`) since the classes are symmetric here; `predict.py --label`
uses it to print REAL/SCREEN alongside the raw score.

## Probability calibration
The winning model (or each component of a winning ensemble) is wrapped in
`CalibratedClassifierCV` (Platt/sigmoid scaling), with calibration itself
using grouped CV so it doesn't leak augmented near-duplicates. I used
`ensemble=False` mode specifically — it fits the base classifier once on
all the data and only cross-fits the small calibration curve, instead of
storing a full separate classifier per CV fold (which would have roughly
tripled the model size for no accuracy benefit). This keeps the saved
model close to the size of a single classifier.

## Error analysis
Every training run now writes to `error_analysis/`: confusion matrix, ROC
curve (with AUC), precision-recall curve, feature-importance bar chart, and
the actual misclassified images (deduplicated by source photo) sorted into
`false_positives/` and `false_negatives/`, filenames prefixed with the
model's confidence score. This is the fastest way to spot systematic
failure modes (e.g. "every misclassified real photo has a bright window in
frame" or "every missed screen photo is a dark-mode display").

## Stronger augmentation
`augment.py` now also includes: small perspective warps, mild affine
shear, slight scale/zoom changes, and gentle exposure shifts (multiplicative,
gentler than plain brightness offset), on top of the original flip /
rotation / brightness-contrast / gamma / blur / JPEG-recompression set —
all still mild enough that the recapture signal in the features survives.

## Data collection — additional diversity to prioritize
Per your point about prioritizing real diversity over augmentation, when
shooting your 50+50:
- **Display types**: OLED phone, LCD/IPS laptop, LCD monitor — each has a
  different pixel structure and moire signature.
- **Brightness**: mix max-brightness and dim screens; mix bright and dark
  content on-screen (dark-mode UI vs. a bright photo).
- **Angle & distance**: not just head-on — oblique angles and varied
  distances change how the FFT/moire and glare features behave.
- **Lighting**: indoor artificial light, daylight, mixed/dim lighting, and
  reflections/glare on the screen glass itself (not just the specular
  highlight case already in the feature set).
- **Real-photo counterpart diversity matters just as much**: shoot real
  objects under the same lighting variety, including some with a TV/
  monitor visible-but-off in the background, so the model doesn't
  spuriously key on "a screen-shaped rectangle is present" rather than "the
  photographed subject *is* a screen."

## Honesty note on validation
I validated this entire pipeline — augmentation, all 41 features, feature
selection, classifier/ensemble comparison, threshold search, calibration,
error-analysis artifacts, and `predict.py` — end-to-end on a synthetic
smoke-test dataset in my sandbox (no camera access there), where it ran
correctly throughout and reached ~99% group-CV accuracy with a 630KB
final model. I could not install XGBoost/LightGBM/torch there (no network
access), so I can't give you a real three-way classifier comparison or a
tested hybrid-CNN number — replace the accuracy/threshold/AUC figures
above with your real numbers once you run `train.py` on your own 50+50
photos.

## Latency & cost
**~40-50 ms per image, warm process, container CPU** for feature
extraction + calibrated-model inference (up from ~35-40ms with the smaller
27-feature version — the multi-scale FFT, JPEG blockiness, and local
contrast additions cost a few ms each, still well within "feels instant").
Model file: **~600KB-1MB** (RandomForest at 200 trees/depth 7, ensemble=False
calibration) — comfortably under the 1MB mobile target. On-device:
effectively free. Cloud fallback: still roughly **$0.01-0.05 per 1,000
images** on a cheap CPU box — the added features don't change this
materially.

## Hybrid CNN (MobileNetV3-Small) — still not the default
`hybrid_cnn.py` is unchanged in spirit: an optional late-fusion experiment,
not the primary solution. With calibration and ensembling now built into
the handcrafted pipeline, the bar for the CNN to clear is higher, not
lower — it would need to beat a calibrated, threshold-tuned, feature-
selected ensemble while still fitting a phone's size/latency budget. My
recommendation stands: keep the handcrafted pipeline as the submission;
only add the CNN if you test it and it clearly, reproducibly wins by a
solid margin.

## Live camera demo
`app.py` + `templates/index.html`: a small local Flask server with a live
camera page. It does **not** reimplement the detector in JavaScript — the
browser just captures a frame every ~900ms and POSTs it to `/predict`,
which runs the exact same `extract_features.py` + `model.pkl` your
`train.py` produced. No image is written to disk or sent anywhere off the
machine it's running on.

```
python app.py
# open http://127.0.0.1:5000  (or this machine's LAN IP, from a phone on the same WiFi)
```

The page shows a live P(screen) meter with the model's optimized cutoff
marked on it, the REAL/SCREEN verdict, per-request latency, and four live
"signal readouts" (JPEG blockiness, glare ratio, FFT high-frequency ratio,
texture regularity) so you can visually see which signals are firing on a
given frame — the explainability of the approach made tangible, not just a
black-box number. If `model.pkl` doesn't exist yet, the page shows an
inline banner telling you to run `augment.py` + `train.py` first, rather
than failing silently. I tested the whole loop — server startup, `/status`,
`/predict` on both a real-style and screen-style image, and the
model-missing banner state — end-to-end in my sandbox with `curl`; I
couldn't test the actual browser camera capture there (no camera/browser
in this environment), so give `getUserMedia` permission a check the first
time you open it locally.

## What I'd improve with more time
- **Screen-bezel/rectangle detection** (Hough-line/contour analysis).
- **Recursive feature elimination (RFE)** as a second feature-selection
  method to cross-check the importance-based pruning used here.
- **Permutation importance** instead of (or alongside) impurity-based
  importance, which can be biased toward high-cardinality features.
- Real XGBoost/LightGBM/hybrid-CNN numbers once run with network access.

## Keeping it accurate as cheaters adapt
- **Feedback loop**: log low-confidence/disputed predictions, get ground
  truth from manual review, retrain regularly — the full pipeline
  (augment → extract → select → compare → calibrate) runs in well under a
  minute even at 1000+ images.
- **Adversarial data collection**: as anti-moire screen protectors, higher
  PPI/refresh-rate screens, or "photo of a printed photo" tricks appear,
  add them to `data/screen/` and rerun.
- **Multi-signal escalation**: for borderline scores, use secondary
  signals the app already has for free — phone depth/parallax API (a
  screen is flat, the real world isn't), burst/multi-frame capture (checks
  for display refresh-rate flicker banding), or EXIF/sensor metadata
  consistency.

## Choosing the cutoff
Automated now (see "Threshold optimization" above), but the deployment
choice of *which* metric to optimize is still a judgment call: given this
is a fraud/cheating context, I'd lean toward the **high-precision
threshold** (favor fewer false accusations of genuine users) for
auto-rejection, and route the wider borderline band to manual review or a
"please retake the photo" nudge rather than a hard block — same policy as
before, now backed by an actual precision/recall curve
(`error_analysis/pr_curve.png`) instead of a guess.