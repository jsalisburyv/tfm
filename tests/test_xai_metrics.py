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


if __name__ == "__main__":
    unittest.main()
