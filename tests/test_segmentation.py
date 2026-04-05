import numpy as np
import unittest

from themlmodel import (
    SegmentationConfig,
    TinySegmentationModel,
    benchmark_model,
    compute_iou,
    evaluate_model,
    train_model,
)


MIN_EXPECTED_TRAIN_IOU = 0.70
MIN_EXPECTED_TRAIN_PIXEL_ACCURACY = 0.90


def _synthetic_dataset(samples: int = 24, size: int = 16, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple segmentation dataset with bright rectangular foregrounds.

    Args:
        samples: Number of image/mask pairs to generate.
        size: Square image side length.
        seed: RNG seed for deterministic generation.

    Returns:
        Tuple of:
            - images shaped [N, 1, H, W] as float64
            - masks shaped [N, H, W] as uint8
    """
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 0.15, size=(samples, 1, size, size))
    masks = np.zeros((samples, size, size), dtype=np.uint8)
    for i in range(samples):
        min_side = max(2, size // 4)
        max_side = max(min_side + 1, size // 3 + 1)
        w = int(rng.integers(min_side, max_side))
        h = int(rng.integers(min_side, max_side))
        x0 = int(rng.integers(0, max(1, size - w + 1)))
        y0 = int(rng.integers(0, max(1, size - h + 1)))
        masks[i, y0 : y0 + h, x0 : x0 + w] = 1
        images[i, 0] += masks[i] * 0.9
    return images.astype(np.float64), masks


class SegmentationTests(unittest.TestCase):
    def test_preprocessing_and_multiclass_output_shape(self):
        images, masks = _synthetic_dataset(samples=4, size=16)
        cfg = SegmentationConfig(epochs=2, pretrain_epochs=1, input_size=64, num_classes=11)
        model = train_model(images, masks, cfg)
        probs = model.predict_proba(images)
        self.assertEqual(probs.shape, (4, 11, 64, 64))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6))

    def test_iou_empty_union_returns_one(self):
        pred = np.zeros((2, 8, 8), dtype=np.float64)
        true = np.zeros((2, 8, 8), dtype=np.uint8)
        self.assertEqual(compute_iou(pred, true), 1.0)

    def test_training_improves_iou(self):
        images, masks = _synthetic_dataset()
        weak_cfg = SegmentationConfig(epochs=1, learning_rate=0.05, pretrain_epochs=1)
        strong_cfg = SegmentationConfig(epochs=20, learning_rate=0.2, pretrain_epochs=10)

        weak_model = train_model(images, masks, weak_cfg)
        strong_model = train_model(images, masks, strong_cfg)

        weak_metrics = evaluate_model(weak_model, images, masks, threshold=0.5)
        strong_metrics = evaluate_model(strong_model, images, masks, threshold=0.5)

        self.assertGreater(strong_metrics["iou"], weak_metrics["iou"])
        self.assertGreater(strong_metrics["iou"], MIN_EXPECTED_TRAIN_IOU)
        self.assertGreater(strong_metrics["pixel_accuracy"], MIN_EXPECTED_TRAIN_PIXEL_ACCURACY)

    def test_benchmark_reports_metric_gains(self):
        images, masks = _synthetic_dataset()
        bench = benchmark_model(
            images,
            masks,
            config=SegmentationConfig(epochs=20, learning_rate=0.2, seed=42, pretrain_epochs=10),
            threshold=0.5,
        )

        self.assertIn("baseline_iou", bench)
        self.assertIn("trained_iou", bench)
        self.assertIn("iou_gain", bench)
        self.assertIn("baseline_pixel_accuracy", bench)
        self.assertIn("trained_pixel_accuracy", bench)
        self.assertIn("pixel_accuracy_gain", bench)

        self.assertGreater(bench["iou_gain"], 0.0)
        self.assertGreater(bench["pixel_accuracy_gain"], 0.0)
        self.assertGreater(bench["trained_iou"], MIN_EXPECTED_TRAIN_IOU)
        self.assertGreater(bench["trained_pixel_accuracy"], MIN_EXPECTED_TRAIN_PIXEL_ACCURACY)

    def test_compute_iou_validates_shape(self):
        pred = np.zeros((2, 8, 8), dtype=np.float64)
        true = np.zeros((2, 8, 7), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_iou(pred, true)

    def test_train_model_validates_input_shapes(self):
        images, masks = _synthetic_dataset(samples=4, size=8)
        with self.assertRaises(ValueError):
            train_model(images[0], masks)
        with self.assertRaises(ValueError):
            train_model(images, masks[..., None])
        with self.assertRaises(ValueError):
            train_model(images, masks[:2])
        with self.assertRaises(ValueError):
            train_model(images, masks[:, :-1, :])

    def test_model_validation_paths(self):
        model = TinySegmentationModel(in_channels=3)
        images, _ = _synthetic_dataset(samples=2, size=8)
        with self.assertRaises(ValueError):
            model.predict_proba(images[0])
        probs = model.predict_proba(images)
        self.assertEqual(probs.ndim, 4)
        with self.assertRaises(ValueError):
            model.set_normalization(np.zeros((2,)), np.ones((2,)))


if __name__ == "__main__":
    unittest.main()
