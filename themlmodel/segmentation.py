from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

EPS = 1e-8
SIGMOID_CLIP_THRESHOLD = 60.0


@dataclass(frozen=True)
class SegmentationConfig:
    learning_rate: float = 0.1
    epochs: int = 200
    threshold: float = 0.5
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -SIGMOID_CLIP_THRESHOLD, SIGMOID_CLIP_THRESHOLD)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class TinySegmentationModel:
    """A minimal from-scratch per-pixel logistic segmentation model."""

    def __init__(self, in_channels: int = 1, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.in_channels = in_channels
        self.weights = np.zeros((in_channels,), dtype=np.float64)
        self.bias = 0.0
        self.channel_mean = np.zeros((in_channels,), dtype=np.float64)
        self.channel_std = np.ones((in_channels,), dtype=np.float64)

    def set_normalization(self, channel_mean: np.ndarray, channel_std: np.ndarray) -> None:
        if channel_mean.shape != (self.in_channels,) or channel_std.shape != (self.in_channels,):
            raise ValueError("normalization shape mismatch")
        self.channel_mean = channel_mean.astype(np.float64)
        self.channel_std = np.maximum(channel_std.astype(np.float64), EPS)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError("images must have shape [N, C, H, W]")
        if images.shape[1] != self.in_channels:
            raise ValueError("channel mismatch for model input")
        normalized = (images - self.channel_mean[None, :, None, None]) / self.channel_std[None, :, None, None]
        logits = np.tensordot(normalized, self.weights, axes=([1], [0])) + self.bias
        return _sigmoid(logits)

    def train_step(self, images: np.ndarray, masks: np.ndarray, lr: float) -> float:
        preds = self.predict_proba(images)
        if masks.ndim != 3:
            raise ValueError("masks must have shape [N, H, W]")
        target = masks.astype(np.float64)
        eps = EPS

        pos = np.sum(target) + eps
        neg = target.size - np.sum(target) + eps
        pos_weight = neg / pos
        sample_weight = np.where(target > 0, pos_weight, 1.0)
        loss = -np.sum(
            sample_weight
            * (target * np.log(preds + eps) + (1.0 - target) * np.log(1.0 - preds + eps))
        ) / target.size

        grad_logits = sample_weight * (preds - target) / np.prod(target.shape)
        normalized = (images - self.channel_mean[None, :, None, None]) / self.channel_std[None, :, None, None]
        grad_w = np.sum(normalized * grad_logits[:, None, :, :], axis=(0, 2, 3))
        grad_b = np.sum(grad_logits)

        self.weights -= lr * grad_w
        self.bias -= lr * grad_b
        return float(loss)


def compute_iou(pred_probs: np.ndarray, true_masks: np.ndarray, threshold: float = 0.5) -> float:
    if pred_probs.shape != true_masks.shape:
        raise ValueError("predictions and masks must have the same shape")
    pred = (pred_probs >= threshold).astype(np.uint8)
    true = (true_masks > 0).astype(np.uint8)

    intersection = np.sum(pred & true, axis=(1, 2)).astype(np.float64)
    union = np.sum(pred | true, axis=(1, 2)).astype(np.float64)
    # If both prediction and target are empty, treat IoU as perfect match (1.0).
    iou_per_sample = np.ones_like(union, dtype=np.float64)
    valid = union > 0.0
    iou_per_sample[valid] = intersection[valid] / union[valid]
    return float(np.mean(iou_per_sample))


def train_model(images: np.ndarray, masks: np.ndarray, config: SegmentationConfig | None = None) -> TinySegmentationModel:
    if config is None:
        config = SegmentationConfig()
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    if images.shape[0] != masks.shape[0] or images.shape[2:] != masks.shape[1:]:
        raise ValueError("image/mask batch or spatial shape mismatch")

    model = TinySegmentationModel(in_channels=images.shape[1], seed=config.seed)
    channel_mean = np.mean(images, axis=(0, 2, 3))
    channel_std = np.std(images, axis=(0, 2, 3))
    model.set_normalization(channel_mean, channel_std)
    for _ in range(config.epochs):
        model.train_step(images, masks, lr=config.learning_rate)
    return model


def evaluate_model(model: TinySegmentationModel, images: np.ndarray, masks: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = model.predict_proba(images)
    iou = compute_iou(probs, masks, threshold=threshold)
    pixel_accuracy = float(np.mean(((probs >= threshold).astype(np.uint8) == (masks > 0).astype(np.uint8))))
    return {"iou": iou, "pixel_accuracy": pixel_accuracy}
