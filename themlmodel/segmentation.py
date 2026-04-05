from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

EPS = 1e-8
SIGMOID_CLIP_THRESHOLD = 60.0
DEFAULT_NUM_CLASSES = 11
DEFAULT_INPUT_SIZE = 256
DEFAULT_TRAIN_EPOCHS = 20
ENCODER_TARGET_SIZE = 64
DECODER_BLEND_WEIGHT = 0.2


@dataclass(frozen=True)
class SegmentationConfig:
    learning_rate: float = 0.01
    epochs: int = DEFAULT_TRAIN_EPOCHS
    threshold: float = 0.5
    seed: int = 42
    input_size: int = DEFAULT_INPUT_SIZE
    num_classes: int = DEFAULT_NUM_CLASSES
    pretrain_epochs: int = 10
    pretrain_learning_rate: float = 0.1


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -SIGMOID_CLIP_THRESHOLD, SIGMOID_CLIP_THRESHOLD)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _binarize_mask(values: np.ndarray, threshold: float, positive_on_nonzero: bool = False) -> np.ndarray:
    if positive_on_nonzero:
        return (values > 0).astype(np.uint8)
    return (values >= threshold).astype(np.uint8)


class TinySegmentationModel:
    """A from-scratch segmentation baseline with UNet-style components."""

    def __init__(self, in_channels: int = 3, num_classes: int = DEFAULT_NUM_CLASSES, input_size: int = DEFAULT_INPUT_SIZE, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.encoder_filters = (64, 128, 256, 512)
        self.decoder_filters = (256, 128, 64, 32)
        self.kernel_size = (3, 3)
        self.pool = 2
        self.channel_mean = np.zeros((in_channels,), dtype=np.float64)
        self.channel_std = np.ones((in_channels,), dtype=np.float64)
        self.class_prototypes = np.zeros((num_classes, in_channels), dtype=np.float64)
        self.class_bias = np.zeros((num_classes,), dtype=np.float64)
        self.rotation_weights = np.zeros((4, 4), dtype=np.float64)
        self.rotation_bias = np.zeros((4,), dtype=np.float64)
        self.rotation_pretrain_accuracy = 0.0

    def set_normalization(self, channel_mean: np.ndarray, channel_std: np.ndarray) -> None:
        if channel_mean.shape != (self.in_channels,) or channel_std.shape != (self.in_channels,):
            raise ValueError("normalization shape mismatch")
        self.channel_mean = channel_mean.astype(np.float64)
        self.channel_std = np.maximum(channel_std.astype(np.float64), EPS)

    def _ensure_rgb(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError("images must have shape [N, C, H, W]")
        if images.shape[1] == self.in_channels:
            return images.astype(np.float64)
        if images.shape[1] == 1 and self.in_channels == 3:
            return np.repeat(images.astype(np.float64), 3, axis=1)
        raise ValueError("channel mismatch for model input")

    def _resize_nearest(self, images: np.ndarray) -> np.ndarray:
        _, _, h, w = images.shape
        if h == self.input_size and w == self.input_size:
            return images.astype(np.float64)
        y_idx = np.clip(np.round(np.linspace(0, h - 1, self.input_size)).astype(np.int64), 0, h - 1)
        x_idx = np.clip(np.round(np.linspace(0, w - 1, self.input_size)).astype(np.int64), 0, w - 1)
        return images[:, :, y_idx][:, :, :, x_idx]

    def preprocess_images(self, images: np.ndarray) -> np.ndarray:
        rgb = self._ensure_rgb(images)
        resized = self._resize_nearest(rgb)
        return (resized - self.channel_mean[None, :, None, None]) / self.channel_std[None, :, None, None]

    def _rotation_features(self, images: np.ndarray) -> np.ndarray:
        half_h = images.shape[2] // 2
        half_w = images.shape[3] // 2
        left = np.mean(images[:, :, :, :half_w], axis=(1, 2, 3))
        right = np.mean(images[:, :, :, half_w:], axis=(1, 2, 3))
        top = np.mean(images[:, :, :half_h, :], axis=(1, 2, 3))
        bottom = np.mean(images[:, :, half_h:, :], axis=(1, 2, 3))
        return np.stack([left - right, top - bottom, left + top, right + bottom], axis=1)

    def self_supervised_pretrain(self, images: np.ndarray, epochs: int, lr: float) -> None:
        preprocessed = self.preprocess_images(images)
        rotated_batches = []
        labels = []
        for angle_idx in range(4):
            rotated_batches.append(np.rot90(preprocessed, k=angle_idx, axes=(2, 3)))
            labels.append(np.full((preprocessed.shape[0],), angle_idx, dtype=np.int64))
        x = np.concatenate(rotated_batches, axis=0)
        y = np.concatenate(labels, axis=0)
        feats = self._rotation_features(x)
        for _ in range(epochs):
            logits = feats @ self.rotation_weights.T + self.rotation_bias[None, :]
            probs = _softmax(logits, axis=1)
            one_hot = np.eye(4)[y]
            grad = (probs - one_hot) / y.shape[0]
            grad_w = grad.T @ feats
            grad_b = np.sum(grad, axis=0)
            self.rotation_weights -= lr * grad_w
            self.rotation_bias -= lr * grad_b
        pred = np.argmax(feats @ self.rotation_weights.T + self.rotation_bias[None, :], axis=1)
        self.rotation_pretrain_accuracy = float(np.mean(pred == y))

    def _avg_pool2x2(self, x: np.ndarray) -> np.ndarray:
        return 0.25 * (x[:, 0::2, 0::2] + x[:, 1::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 1::2])

    def _upsample2x(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)

    def _conv3_like(self, x: np.ndarray) -> np.ndarray:
        pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        return (
            pad[:, :-2, :-2]
            + pad[:, :-2, 1:-1]
            + pad[:, :-2, 2:]
            + pad[:, 1:-1, :-2]
            + pad[:, 1:-1, 1:-1]
            + pad[:, 1:-1, 2:]
            + pad[:, 2:, :-2]
            + pad[:, 2:, 1:-1]
            + pad[:, 2:, 2:]
        ) / 9.0

    def _encode(self, normalized: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        # Keep encoder lightweight by collapsing RGB channels to one feature map.
        x = np.mean(normalized, axis=1)
        skips: list[np.ndarray] = []
        for _ in self.encoder_filters:
            x = self._conv3_like(x)
            skips.append(x)
            x = self._avg_pool2x2(x)
        return x, skips

    def _apply_attention(self, features: list[np.ndarray]) -> list[np.ndarray]:
        attended = []
        for feat in features:
            channel_att = np.mean(np.abs(feat), axis=(1, 2), keepdims=True)
            channel_att = channel_att / (channel_att + 1.0)
            spatial_att = np.abs(feat)
            spatial_att = spatial_att / (np.max(spatial_att, axis=(1, 2), keepdims=True) + EPS)
            attended.append(feat * (0.5 + 0.5 * channel_att) * (0.5 + 0.5 * spatial_att))
        return attended

    def _decode(self, bottleneck: np.ndarray, skips: list[np.ndarray]) -> np.ndarray:
        x = bottleneck
        for idx, _ in enumerate(self.decoder_filters):
            x = self._upsample2x(x)
            skip = skips[-(idx + 1)]
            if x.shape != skip.shape:
                x = x[:, : skip.shape[1], : skip.shape[2]]
            x = self._conv3_like(0.5 * (x + skip))
        return x[:, : self.input_size, : self.input_size]

    def _decoder_map(self, normalized: np.ndarray) -> np.ndarray:
        stride = max(1, self.input_size // ENCODER_TARGET_SIZE)
        downsampled = normalized[:, :, ::stride, ::stride]
        bottleneck, skips = self._encode(downsampled)
        attended_skips = self._apply_attention(skips)
        decoded = self._decode(bottleneck, attended_skips)
        upsampled = np.repeat(np.repeat(decoded, stride, axis=1), stride, axis=2)
        return upsampled[:, : self.input_size, : self.input_size]

    def _compute_logits(self, normalized: np.ndarray) -> np.ndarray:
        decoder_map = self._decoder_map(normalized)
        diff = normalized[:, None, :, :, :] - self.class_prototypes[None, :, :, None, None]
        color_score = -np.sum(diff * diff, axis=2)
        return color_score + self.class_bias[None, :, None, None] + DECODER_BLEND_WEIGHT * decoder_map[:, None, :, :]

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        normalized = self.preprocess_images(images)
        logits = self._compute_logits(normalized)
        return _softmax(logits, axis=1)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(images), axis=1).astype(np.uint8)

    def train_step(self, images: np.ndarray, masks: np.ndarray, lr: float) -> float:
        if masks.ndim != 3:
            raise ValueError("masks must have shape [N, H, W]")
        normalized = self.preprocess_images(images)
        resized_masks = _resize_mask_nearest(masks, self.input_size)
        probs = self.predict_proba(images)
        one_hot = np.eye(self.num_classes, dtype=np.float64)[resized_masks]
        class_counts = np.sum(one_hot, axis=(0, 1, 2)) + EPS
        prototype_update = np.sum(
            normalized[:, None, :, :, :] * one_hot.transpose(0, 3, 1, 2)[:, :, None, :, :],
            axis=(0, 3, 4),
        ) / class_counts[:, None]
        self.class_prototypes = (1.0 - lr) * self.class_prototypes + lr * prototype_update
        class_freq = class_counts / np.sum(class_counts)
        bias_target = np.log(np.maximum(class_freq, EPS))
        self.class_bias = (1.0 - lr) * self.class_bias + lr * bias_target
        loss = -np.mean(np.sum(one_hot * np.log(np.maximum(probs.transpose(0, 2, 3, 1), EPS)), axis=-1))
        return float(loss)


def compute_iou(pred_probs: np.ndarray, true_masks: np.ndarray, threshold: float = 0.5) -> float:
    """Compute mean IoU across a batch of predicted probabilities and masks.

    Args:
        pred_probs: Array shaped [N, H, W] with per-pixel probabilities.
        true_masks: Array shaped [N, H, W] with binary ground-truth masks.
        threshold: Probability cutoff used to binarize predictions.

    Returns:
        Mean IoU across samples. Samples with empty prediction and target union
        are treated as perfect matches and assigned IoU=1.0.
    """
    if pred_probs.ndim == 3:
        if pred_probs.shape != true_masks.shape:
            raise ValueError("predictions and masks must have the same shape")
        pred = _binarize_mask(pred_probs, threshold=threshold)
        true = _binarize_mask(true_masks, threshold=threshold, positive_on_nonzero=True)
        intersection = np.sum(pred & true, axis=(1, 2)).astype(np.float64)
        union = np.sum(pred | true, axis=(1, 2)).astype(np.float64)
        iou_per_sample = np.ones_like(union, dtype=np.float64)
        valid = union > 0.0
        iou_per_sample[valid] = intersection[valid] / union[valid]
        return float(np.mean(iou_per_sample))
    if pred_probs.ndim != 4:
        raise ValueError("predictions must have shape [N, H, W] or [N, C, H, W]")
    if true_masks.ndim != 3 or pred_probs.shape[0] != true_masks.shape[0] or pred_probs.shape[2:] != true_masks.shape[1:]:
        raise ValueError("multiclass predictions and masks must align on [N, H, W]")
    pred = np.argmax(pred_probs, axis=1).astype(np.int64)
    true = true_masks.astype(np.int64)
    num_classes = pred_probs.shape[1]
    sample_ious = []
    for i in range(pred.shape[0]):
        class_ious = []
        for cls in range(num_classes):
            p = pred[i] == cls
            t = true[i] == cls
            union = np.sum(p | t)
            if union == 0:
                continue
            class_ious.append(float(np.sum(p & t) / union))
        sample_ious.append(1.0 if not class_ious else float(np.mean(class_ious)))
    return float(np.mean(sample_ious))


def train_model(images: np.ndarray, masks: np.ndarray, config: SegmentationConfig | None = None) -> TinySegmentationModel:
    """Train a tiny from-scratch segmentation model.

    Args:
        images: Input batch shaped [N, C, H, W].
        masks: Binary target masks shaped [N, H, W].
        config: Optional training hyperparameters.

    Returns:
        A trained TinySegmentationModel.
    """
    if config is None:
        config = SegmentationConfig()
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    if images.shape[0] != masks.shape[0] or images.shape[2:] != masks.shape[1:]:
        raise ValueError("image/mask batch or spatial shape mismatch")
    if np.any(masks < 0):
        raise ValueError("masks cannot contain negative class ids")

    model = TinySegmentationModel(in_channels=3, num_classes=config.num_classes, input_size=config.input_size, seed=config.seed)
    rgb_images = model._ensure_rgb(images)
    resized = model._resize_nearest(rgb_images)
    channel_mean = np.mean(resized, axis=(0, 2, 3))
    channel_std = np.std(resized, axis=(0, 2, 3))
    model.set_normalization(channel_mean, channel_std)
    model.self_supervised_pretrain(images, epochs=config.pretrain_epochs, lr=config.pretrain_learning_rate)
    for _ in range(config.epochs):
        model.train_step(images, masks, lr=config.learning_rate)
    return model


def evaluate_model(model: TinySegmentationModel, images: np.ndarray, masks: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate segmentation performance with IoU and pixel accuracy.

    Args:
        model: Trained segmentation model.
        images: Input batch shaped [N, C, H, W].
        masks: Binary target masks shaped [N, H, W].
        threshold: Probability threshold for binarizing predictions.

    Returns:
        Dictionary containing:
            - "iou": mean intersection-over-union
            - "pixel_accuracy": per-pixel classification accuracy
    """
    probs = model.predict_proba(images)
    resized_masks = _resize_mask_nearest(masks, model.input_size)
    iou = compute_iou(probs, resized_masks, threshold=threshold)
    pred = np.argmax(probs, axis=1)
    pixel_accuracy = float(np.mean(pred == resized_masks))
    return {"iou": iou, "pixel_accuracy": pixel_accuracy}


def benchmark_model(
    images: np.ndarray,
    masks: np.ndarray,
    config: SegmentationConfig | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Benchmark untrained baseline vs trained model on the same dataset."""
    if config is None:
        config = SegmentationConfig()
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    if images.shape[0] != masks.shape[0] or images.shape[2:] != masks.shape[1:]:
        raise ValueError("image/mask batch or spatial shape mismatch")

    preprocessing_model = TinySegmentationModel(
        in_channels=3,
        num_classes=config.num_classes,
        input_size=config.input_size,
        seed=config.seed,
    )
    rgb_images = preprocessing_model._ensure_rgb(images)
    resized = preprocessing_model._resize_nearest(rgb_images)
    channel_mean = np.mean(resized, axis=(0, 2, 3))
    channel_std = np.std(resized, axis=(0, 2, 3))

    baseline_model = TinySegmentationModel(in_channels=3, num_classes=config.num_classes, input_size=config.input_size, seed=config.seed)
    baseline_model.set_normalization(channel_mean, channel_std)
    baseline_metrics = evaluate_model(baseline_model, images, masks, threshold=threshold)

    trained_model = train_model(images, masks, config=config)
    trained_metrics = evaluate_model(trained_model, images, masks, threshold=threshold)

    return {
        "baseline_iou": baseline_metrics["iou"],
        "baseline_pixel_accuracy": baseline_metrics["pixel_accuracy"],
        "trained_iou": trained_metrics["iou"],
        "trained_pixel_accuracy": trained_metrics["pixel_accuracy"],
        "iou_gain": trained_metrics["iou"] - baseline_metrics["iou"],
        "pixel_accuracy_gain": trained_metrics["pixel_accuracy"] - baseline_metrics["pixel_accuracy"],
    }


def _resize_mask_nearest(masks: np.ndarray, size: int) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    _, h, w = masks.shape
    if h == size and w == size:
        return masks.astype(np.int64)
    y_idx = np.clip(np.round(np.linspace(0, h - 1, size)).astype(np.int64), 0, h - 1)
    x_idx = np.clip(np.round(np.linspace(0, w - 1, size)).astype(np.int64), 0, w - 1)
    return masks[:, y_idx][:, :, x_idx].astype(np.int64)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(np.clip(shifted, -SIGMOID_CLIP_THRESHOLD, SIGMOID_CLIP_THRESHOLD))
    return exps / (np.sum(exps, axis=axis, keepdims=True) + EPS)
