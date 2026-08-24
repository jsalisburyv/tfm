"""Compute and cache Sensitivity-N for the 30-video evaluation subset.

This is intentionally separate from the analysis notebook because it performs
model inference. Run a smoke test first, for example:

    python tools/run_sensitivity_n.py --datasets EtriActivity3D --limit-videos 1 --samples 8 --n 3

The full default is N=10, 100 sampled subsets, seed=777. Existing rows are
resumed unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "video_xai"))
sys.path.insert(0, str(PROJECT_ROOT / "mmaction2"))

from mmcv import Config
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.datasets import PIPELINES
from mmaction.datasets.pipelines import Compose

from utils import load_video
from xai_metrics import SensitivityNEvaluator


MODEL = "TANet"
METHODS = [
    "3D-Kernel-SHAP-NEW",
    "3D-LIME-NEW",
    "3D-RISE-NEW",
    "3D-Sampled-Occl-Sens-NEW",
    "LV-LOCO-NEW",
    "LV-Univ-Pred-NEW",
    "SaliencyTubes",
    "AOSA",
    "GradCAM",
]
SLIC_METHODS = {
    "3D-Kernel-SHAP-NEW",
    "3D-LIME-NEW",
    "LV-LOCO-NEW",
    "LV-Univ-Pred-NEW",
}
N_CLASSES = {"EtriActivity3D": 55, "Kinetics400": 400}


@PIPELINES.register_module(force=True)
class SensitivityArrayTransform:
    """MMACTION transform matching the array inference path in notebook 06."""

    def __init__(self, clip_len=1, frame_interval=1, num_clips=1):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def __call__(self, results):
        images = [results["array"][frame] for frame in range(results["array"].shape[0])]
        results["imgs"] = images
        results["original_shape"] = images[0].shape[:2]
        results["img_shape"] = images[0].shape[:2]
        results["clip_len"] = self.clip_len
        results["frame_interval"] = self.frame_interval
        results["num_clips"] = self.num_clips
        return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(N_CLASSES), default=sorted(N_CLASSES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--n", type=int, default=10, help="Regions in each subset (default: 10)")
    parser.add_argument("--samples", type=int, default=100, help="Sampled subsets (default: 100)")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-videos", type=int, default=None, help="Per-dataset smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "results" / "sensitivity_n_per_video.csv",
    )
    return parser.parse_args()


def load_zipped_npy(path):
    with ZipFile(path) as archive:
        with archive.open(archive.namelist()[0]) as file:
            return np.load(file)


def normalise_heatmap(score_map):
    heatmap = np.nan_to_num(np.asarray(score_map, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap -= heatmap.min()
    maximum = heatmap.max()
    return heatmap / maximum if maximum > 0 else heatmap


def indexed_scores_to_map(scores, shape, invert=False):
    values = np.asarray(scores[:, 1], dtype=np.float32)
    if invert:
        values = 1.0 - values
    needed = int(np.prod(shape))
    if len(values) < needed:
        values = np.pad(values, (0, needed - len(values)), constant_values=np.nan)
    return values[:needed].reshape(shape)


def load_score_and_regions(generated_dir, dataset, video_class, video_name, method):
    sample_dir = generated_dir / dataset / "data-labels" / MODEL / video_class / video_name
    scores = np.load(sample_dir / f"{method}-scores.npy", allow_pickle=True)
    region_map = None

    if method in SLIC_METHODS:
        segment_path = (
            generated_dir
            / dataset
            / "segments"
            / video_class
            / f"{Path(video_name).stem}_80x256x256.zip"
        )
        segments = load_zipped_npy(segment_path)
        ids = scores[:, 0].astype(int)
        values = scores[:, 1].astype(np.float32)
        values = np.maximum(values, 0.0)
        lookup = np.zeros(max(int(segments.max()), int(ids.max())) + 1, dtype=np.float32)
        lookup[ids] = values
        score_map = lookup[segments]
        region_map = segments
    elif method == "3D-RISE-NEW":
        score_map = indexed_scores_to_map(scores, (4, 7, 7))
    elif method == "3D-Sampled-Occl-Sens-NEW":
        # Matches ConvolutionalVisualizer(kernel_size=[7,4,4], stride=[2,2,2])
        # used by 06-evaluation.ipynb.
        score_map = indexed_scores_to_map(scores, (13, 7, 7), invert=True)
    else:
        score_map = scores

    return normalise_heatmap(score_map), region_map


def sample_frames(video, pipeline):
    result = pipeline(
        dict(total_frames=video.shape[0], label=-1, start_index=0, array=video, modality="RGB")
    )
    ordered = sorted(enumerate(result["frame_inds"]), key=lambda pair: pair[1])
    sorted_indices, sorted_frames = zip(*ordered)
    return sorted_indices, video[list(sorted_frames)]


def unsort_frames(sorted_indices, frames):
    unsorted_indices, _ = zip(*sorted(zip(range(frames.shape[0]), sorted_indices), key=lambda pair: pair[1]))
    return frames[list(unsorted_indices)]


def model_context(dataset, generated_dir, device):
    config_path = (
        PROJECT_ROOT
        / "mmaction2"
        / "configs"
        / "recognition"
        / "tanet"
        / "tanet_r50_dense_1x1x8_100e_kinetics400_rgb - modified pipeline.py"
    )
    weight_name = (
        "tanet_r50_dense_1x1x8_etri_epoch10_finetuned.pth"
        if dataset == "EtriActivity3D"
        else "tanet_r50_dense_1x1x8_kinetics400_pretrained_20210219-032c8e94.pth"
    )
    weight_path = PROJECT_ROOT / "artifacts" / "model-weights" / weight_name

    cfg = Config.fromfile(str(config_path))
    cfg.model.cls_head.num_classes = N_CLASSES[dataset]
    cfg.data.test.data_prefix = str(generated_dir / dataset / "videos")
    model = init_recognizer(cfg, str(weight_path), device=device)

    frame_config = model.cfg.data.test.pipeline[1]
    frame_pipeline = Compose([frame_config])
    cfg.data.test.pipeline[2] = dict(
        type="SensitivityArrayTransform",
        clip_len=frame_config["clip_len"],
        frame_interval=frame_config["frame_interval"],
        num_clips=frame_config["num_clips"],
    )
    cfg.data.test.pipeline.pop(1)
    return model, frame_pipeline


def read_existing(path):
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    keys = {(row["dataset"], row["video_class"], row["video_name"], row["method"]) for row in rows}
    return rows, keys


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "model", "method", "video_class", "video_name", "sensitivity_n",
        "random_correlation", "reversed_correlation", "n", "n_subsets", "seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    generated_dir = PROJECT_ROOT / "artifacts" / "generated-explanations"
    rows, completed = ([], set()) if args.overwrite else read_existing(args.output)

    for dataset in args.datasets:
        print(f"Loading {MODEL} for {dataset} on {args.device}")
        model, frame_pipeline = model_context(dataset, generated_dir, args.device)

        video_paths = sorted((generated_dir / dataset / "videos_small").glob("*/*"))
        if args.limit_videos is not None:
            video_paths = video_paths[: args.limit_videos]

        for video_path in video_paths:
            video_class = video_path.parent.name
            video_name = video_path.name
            video = load_video(str(video_path))
            sorted_indices, frames_in = sample_frames(video, frame_pipeline)

            def classifier(videos, *, _model=model, _indices=sorted_indices):
                predictions = [inference_recognizer(_model, unsort_frames(_indices, item)) for item in videos]
                return np.asarray(predictions)

            for method in args.methods:
                key = (dataset, video_class, video_name, method)
                if key in completed:
                    continue
                print(f"  {video_class}/{video_name}: {method}")
                score_map, region_map = load_score_and_regions(
                    generated_dir, dataset, video_class, video_name, method
                )
                evaluator = SensitivityNEvaluator(
                    frames_in,
                    score_map,
                    classifier,
                    label=int(video_class),
                    region_map=region_map,
                    n_regions=args.n,
                    n_subsets=args.samples,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    hide_color="blur",
                    blur_mode="3d",
                    blur_radius=25,
                )
                score = evaluator.evaluate()
                checks = evaluator.sanity_checks()
                rows.append(
                    {
                        "dataset": dataset,
                        "model": MODEL,
                        "method": method,
                        "video_class": video_class,
                        "video_name": video_name,
                        "sensitivity_n": score,
                        **checks,
                        "n": args.n,
                        "n_subsets": args.samples,
                        "seed": args.seed,
                    }
                )
                completed.add(key)
                write_rows(args.output, rows)

    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
