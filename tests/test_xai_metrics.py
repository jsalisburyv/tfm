from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "video_xai"))

from xai_metrics import SensitivityNEvaluator, regular_grid_regions


class SensitivityNTests(unittest.TestCase):
    def test_regular_grid_regions_has_expected_shape_and_labels(self):
        regions = regular_grid_regions((4, 6, 8), (2, 3, 4))
        self.assertEqual(regions.shape, (4, 6, 8))
        self.assertEqual(len(np.unique(regions)), 24)


    def test_sensitivity_n_is_deterministic_and_has_correct_direction(self):
        regions = regular_grid_regions((2, 4, 4), (1, 2, 2))
        important_region = int(regions[0, 0, 0])
        video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        video[regions == important_region] = 255
        heatmap = (regions == important_region).astype(np.float32)

        def classifier(videos):
            target = videos[..., 0].mean(axis=(1, 2, 3)) / 255.0
            return np.column_stack([target, 1.0 - target])

        kwargs = dict(
            video=video,
            score_map=heatmap,
            classifier_fn=classifier,
            label=0,
            region_map=regions,
            n_regions=2,
            n_subsets=6,
            seed=7,
            hide_color=0,
            batch_size=3,
        )
        first = SensitivityNEvaluator(**kwargs)
        second = SensitivityNEvaluator(**kwargs)

        self.assertGreater(first.evaluate(), 0.99)
        self.assertEqual(first.score, second.evaluate())
        checks = first.sanity_checks()
        self.assertGreater(first.score, checks["random_correlation"])
        self.assertLess(checks["reversed_correlation"], -0.99)


    def test_sensitivity_n_returns_nan_for_constant_attribution(self):
        video = np.ones((2, 4, 4, 3), dtype=np.uint8)

        def classifier(videos):
            target = videos[..., 0].mean(axis=(1, 2, 3))
            return np.column_stack([target, target])

        evaluator = SensitivityNEvaluator(
            video,
            np.ones(video.shape[:3], dtype=np.float32),
            classifier,
            label=0,
            grid_shape=(1, 2, 2),
            n_regions=2,
            n_subsets=6,
            hide_color=0,
        )
        self.assertTrue(np.isnan(evaluator.evaluate()))

    def test_compatible_explanations_reuse_perturbation_predictions(self):
        regions = regular_grid_regions((2, 4, 4), (1, 2, 2))
        video = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        classifier_calls = []

        def classifier(videos):
            classifier_calls.append(len(videos))
            target = videos[..., 0].mean(axis=(1, 2, 3)) / 255.0
            return np.column_stack([target, 1.0 - target])

        common = dict(
            video=video,
            classifier_fn=classifier,
            label=0,
            region_map=regions,
            n_regions=2,
            n_subsets=6,
            seed=7,
            hide_color=0,
            batch_size=3,
        )
        source = SensitivityNEvaluator(
            score_map=(regions == 0).astype(np.float32),
            **common,
        )
        source.evaluate()
        calls_after_source = list(classifier_calls)

        reused = SensitivityNEvaluator(
            score_map=(regions == 1).astype(np.float32),
            **common,
        )
        reused.evaluate(perturbation_source=source)
        calls_after_reuse = list(classifier_calls)

        independent = SensitivityNEvaluator(
            score_map=(regions == 1).astype(np.float32),
            **common,
        )
        independent_score = independent.evaluate()

        self.assertEqual(calls_after_reuse, calls_after_source)
        np.testing.assert_array_equal(reused.subsets, source.subsets)
        np.testing.assert_allclose(reused.output_changes, source.output_changes)
        self.assertEqual(reused.score, independent_score)

    def test_incompatible_regions_cannot_reuse_perturbations(self):
        video = np.zeros((2, 4, 4, 3), dtype=np.uint8)

        def classifier(videos):
            target = videos[..., 0].mean(axis=(1, 2, 3))
            return np.column_stack([target, target])

        common = dict(
            video=video,
            score_map=np.ones(video.shape[:3], dtype=np.float32),
            classifier_fn=classifier,
            label=0,
            n_regions=2,
            n_subsets=6,
            hide_color=0,
        )
        source = SensitivityNEvaluator(
            region_map=regular_grid_regions(video.shape[:3], (1, 2, 2)),
            **common,
        )
        source.evaluate()
        incompatible = SensitivityNEvaluator(
            region_map=regular_grid_regions(video.shape[:3], (2, 1, 2)),
            **common,
        )

        with self.assertRaisesRegex(ValueError, "different region map"):
            incompatible.evaluate(perturbation_source=source)


if __name__ == "__main__":
    unittest.main()
