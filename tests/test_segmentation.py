import unittest

import numpy as np

from themlmodel import SegmentationConfig, compute_iou, evaluate_model, train_model


def _synthetic_dataset(samples: int = 24, size: int = 16, seed: int = 3):
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 0.15, size=(samples, 1, size, size))
    masks = np.zeros((samples, size, size), dtype=np.uint8)
    for i in range(samples):
        x0 = int(rng.integers(2, size // 2))
        y0 = int(rng.integers(2, size // 2))
        w = int(rng.integers(size // 4, size // 3))
        h = int(rng.integers(size // 4, size // 3))
        masks[i, y0 : y0 + h, x0 : x0 + w] = 1
        images[i, 0] += masks[i] * 0.9
    return images.astype(np.float64), masks


class SegmentationTests(unittest.TestCase):
    def test_iou_empty_union_returns_one(self):
        pred = np.zeros((2, 8, 8), dtype=np.float64)
        true = np.zeros((2, 8, 8), dtype=np.uint8)
        self.assertEqual(compute_iou(pred, true), 1.0)

    def test_training_improves_iou(self):
        images, masks = _synthetic_dataset()
        weak_cfg = SegmentationConfig(epochs=1, learning_rate=0.01)
        strong_cfg = SegmentationConfig(epochs=250, learning_rate=0.1)

        weak_model = train_model(images, masks, weak_cfg)
        strong_model = train_model(images, masks, strong_cfg)

        weak_metrics = evaluate_model(weak_model, images, masks, threshold=0.5)
        strong_metrics = evaluate_model(strong_model, images, masks, threshold=0.5)

        self.assertGreater(strong_metrics["iou"], weak_metrics["iou"])
        self.assertGreater(strong_metrics["iou"], 0.80)
        self.assertGreater(strong_metrics["pixel_accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
