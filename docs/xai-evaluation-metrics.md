# XAI evaluation metrics

## Sensitivity-N

Category: `sensitivity`. Direction: higher is better.

For each video/method, sample 100 unique subsets of 10 input regions with seed
777. Sum the mean attribution of the selected regions, replace those regions by
the REVEX 3-D Gaussian-blur baseline (radius 25), and measure the target-class
score drop. Sensitivity-N is the Pearson correlation between attribution sums
and score drops. SLIC explanations reuse their saved segmentation; the other
methods use a common 4x7x7 spatiotemporal grid. This requires 101 target-model
predictions per video/method (one original plus 100 perturbations), but never
recomputes an explanation.

NaN is returned for constant attributions or constant model responses. The
cache stores correlations for deterministic random and reversed region scores
using the same perturbations. Partial smoke-test results do not enter rankings.

Run:

```powershell
# Small compatibility test
python tools/run_sensitivity_n.py --datasets EtriActivity3D --limit-videos 1 --samples 8 --n 3

# Full configured evaluation (resumable)
python tools/run_sensitivity_n.py
```

## ROAR

Category: `retraining`. Direction: higher accuracy degradation is better.

The intended protocol removes the most relevant 10%, 30%, and 50% of regions
from the training data, retrains TANet from the same initialization, and
evaluates on the correspondingly transformed held-out data. The cache stores
both post-retraining accuracy and `baseline_accuracy - post_retraining_accuracy`.
The notebook averages the latter across the three removal levels while retaining
the raw dataset/method/fraction rows. Every run must include a matched random-
removal control.

ROAR is currently unavailable because the original training videos, annotation
splits, and full-training explanation maps are absent. See
`docs/roar-feasibility-audit.md`. Blur-induced distribution shift remains a
limitation even after the missing artifacts are restored, so guided removal
must be interpreted against random removal.

## Taxonomy boundary

Deletion AUC, Insertion AUC, and Average Drop are input-perturbation metrics;
Pointing Game and IoU are ground-truth-comparison metrics. Runtime, temporal
consistency, sparsity, and bounding-box energy ratio remain explicitly outside
the Kadir categories used here.

