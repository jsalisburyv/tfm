"""ROAR preflight and cached-result validation.

Full ROAR is deliberately not launched here. The repository does not contain
the original training videos/split files or training-set explanations. This
tool makes those prerequisites explicit and validates externally produced ROAR
results before the analysis notebook consumes them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "artifacts" / "generated-explanations"
METHODS = [
    "3D-Kernel-SHAP-NEW", "3D-LIME-NEW", "3D-RISE-NEW",
    "3D-Sampled-Occl-Sens-NEW", "LV-LOCO-NEW", "LV-Univ-Pred-NEW",
    "SaliencyTubes", "AOSA", "GradCAM",
]
DATASETS = ["EtriActivity3D", "Kinetics400"]
REQUIRED_RESULT_COLUMNS = {
    "dataset", "model", "method", "removal_fraction", "baseline_accuracy",
    "post_retraining_accuracy", "accuracy_drop", "random_baseline_accuracy",
    "random_accuracy_drop", "seed", "config_path", "checkpoint_path", "status",
}


def audit():
    report = {
        "feasible": False,
        "training_entry_point": "mmaction2/tools/train.py and mmaction.apis.train_model (used in notebooks/02-training.ipynb)",
        "training_notebook": "notebooks/02-training.ipynb",
        "datasets": {},
        "planned_removal_fractions": [0.1, 0.3, 0.5],
        "guided_retraining_runs": len(DATASETS) * len(METHODS) * 3,
        "random_baseline_retraining_runs": len(DATASETS) * 3,
        "total_retraining_runs_per_seed": len(DATASETS) * (len(METHODS) + 1) * 3,
        "blockers": [
            "Original training videos are external and are not present under artifacts/generated-explanations.",
            "The exact annotation/split files referenced by the training notebook are external Z:/ and D:/ paths.",
            "Saved explanations cover only the selected 30 evaluation videos per dataset, not training samples.",
            "Full training-set explanations must be generated before explanation-guided removal can be applied without leakage.",
        ],
    }

    for dataset in DATASETS:
        video_count = len(list((GENERATED_DIR / dataset / "videos_small").glob("*/*")))
        method_counts = {}
        labels_root = GENERATED_DIR / dataset / "data-labels" / "TANet"
        for method in METHODS:
            method_counts[method] = len(list(labels_root.glob(f"*/*/{method}-scores.npy")))
        report["datasets"][dataset] = {
            "selected_evaluation_videos": video_count,
            "saved_explanations_by_method": method_counts,
            "training_annotation_files_in_repository": [],
            "training_videos_in_repository": 0,
        }
    return report


def validate_results(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_RESULT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing ROAR columns: {sorted(missing)}")
        rows = list(reader)

    seen = set()
    for row_number, row in enumerate(rows, start=2):
        key = (row["dataset"], row["method"], row["removal_fraction"], row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate ROAR run at row {row_number}: {key}")
        seen.add(key)
        if row["status"] != "complete":
            continue
        fraction = float(row["removal_fraction"])
        if fraction not in {0.1, 0.3, 0.5}:
            raise ValueError(f"Unexpected removal fraction at row {row_number}: {fraction}")
        baseline = float(row["baseline_accuracy"])
        retrained = float(row["post_retraining_accuracy"])
        drop = float(row["accuracy_drop"])
        if abs((baseline - retrained) - drop) > 1e-8:
            raise ValueError(f"accuracy_drop mismatch at row {row_number}")
        if not row["random_baseline_accuracy"] or not row["random_accuracy_drop"]:
            raise ValueError(f"Random-removal baseline missing at row {row_number}")
        if Path(row["config_path"]).name == "" or Path(row["checkpoint_path"]).name == "":
            raise ValueError(f"Config/checkpoint provenance missing at row {row_number}")
    return {"rows": len(rows), "complete_rows": sum(row["status"] == "complete" for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path)
    validate_parser = subparsers.add_parser("validate-results")
    validate_parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if args.command == "audit":
        result = audit()
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    else:
        print(json.dumps(validate_results(args.path), indent=2))


if __name__ == "__main__":
    main()

