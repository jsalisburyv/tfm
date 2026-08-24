# ROAR feasibility audit

Status: **blocked by missing training artifacts; no ROAR quality values have
been fabricated.**

## What is reproducible

TANet training is called through `mmaction.apis.train_model` in
`notebooks/02-training.ipynb`; the generic CLI entry point also exists at
`mmaction2/tools/train.py`. The notebook records 10-epoch fine-tuning, SGD,
batch size 2, learning rate divided by eight, and seed 777 for EtriActivity3D.
The repository contains the fine-tuned ETRI checkpoint and the public
Kinetics400-pretrained TANet checkpoint.

The evaluation path is reproducible for the selected 30 videos per dataset.
It fixes the target to the ground-truth class and uses Gaussian blur
(`blur_mode="3d"`, radius 25), which is also used by Sensitivity-N.

## Blocking artifacts

- The ETRI and Kinetics training videos are not in the repository.
- The training and validation annotation files referenced by the notebooks are
  hard-coded to external `Z:/` and `D:/` locations and are not present here.
- Exact Kinetics pretraining data order and initialization provenance are not
  packaged locally.
- Explanations exist for the 30 selected evaluation videos only. Each of the
  nine methods would need explanations for every training sample.

Consequently, modifying the 30 evaluation videos and fine-tuning on them would
be data leakage and is not ROAR. The full experiment must remain disabled until
the original split files, training videos, and full-training explanation maps
are supplied.

## Planned protocol once unblocked

For each dataset, explanation method, and removal fraction (10%, 30%, 50%),
rank regions, replace the selected regions with the same Gaussian-blurred
baseline, retrain TANet from the same initialization/configuration, and evaluate
on the correspondingly transformed held-out split. Store raw accuracy and
`baseline_accuracy - post_retraining_accuracy`; the latter is the higher-is-
better `roar` ranking value. Run matched random-region removal for every
fraction and seed. Keep training explanations and validation evaluation
strictly separated.

One seed requires 54 guided retraining runs and 6 random controls: 60 complete
TANet training/evaluation runs, plus nine explanation passes over each training
set. Wall-clock cost cannot be estimated reliably without the missing dataset
sizes, explanation timings on training data, and target hardware; it is best
reported as 60 training-run equivalents per seed.

`tools/roar_pipeline.py audit` produces a machine-readable preflight report.
`tools/roar_pipeline.py validate-results docs/results/roar_results.csv` checks
the cache schema, arithmetic, provenance, duplicate runs, and presence of the
random baseline before notebook ingestion.

Distribution shift is an important limitation: even with random controls, a
drop may partly reflect the blur/removal artifact rather than attribution
quality. Conclusions should therefore compare guided removal against matched
random removal, not interpret raw degradation alone.

