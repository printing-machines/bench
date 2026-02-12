"""
PaintBench Evaluation Metrics

Computes similarity metrics between the final canvas and target image:
- CLIP cosine similarity (artistic/semantic similarity)
- LPIPS (learned perceptual similarity)
- SSIM (structural similarity)
- MSE (pixel-level accuracy)
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

# Lazy imports for optional dependencies
_clip_model: Any = None
_clip_preprocess: Callable[..., torch.Tensor] | None = None
_lpips_model: Any = None


def _load_clip():
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    return _clip_model, _clip_preprocess


def _load_lpips():
    """Lazy load LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        import lpips

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_model = lpips.LPIPS(net="alex").to(device)
    return _lpips_model


def compute_mse(canvas: Image.Image, target: Image.Image) -> float:
    """Compute Mean Squared Error between images."""
    canvas_np = np.array(canvas.convert("RGB")).astype(np.float32) / 255.0
    target_np = np.array(target.convert("RGB")).astype(np.float32) / 255.0
    return float(np.mean((canvas_np - target_np) ** 2))


def compute_ssim(canvas: Image.Image, target: Image.Image) -> float:
    """Compute Structural Similarity Index."""
    from skimage.metrics import structural_similarity

    canvas_np = np.array(canvas.convert("RGB"))
    target_np = np.array(target.convert("RGB"))

    return float(structural_similarity(canvas_np, target_np, channel_axis=2, data_range=255))  # type: ignore[arg-type]


def compute_clip_similarity(canvas: Image.Image, target: Image.Image) -> float:
    """Compute CLIP cosine similarity between images."""
    model, preprocess = _load_clip()
    assert preprocess is not None
    device = next(model.parameters()).device

    canvas_tensor = preprocess(canvas).unsqueeze(0).to(device)  # type: ignore[union-attr]
    target_tensor = preprocess(target).unsqueeze(0).to(device)  # type: ignore[union-attr]

    with torch.no_grad():
        canvas_features = model.encode_image(canvas_tensor)
        target_features = model.encode_image(target_tensor)

        # Normalize
        canvas_features = canvas_features / canvas_features.norm(dim=-1, keepdim=True)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (canvas_features @ target_features.T).item()

    return float(similarity)


def compute_lpips(canvas: Image.Image, target: Image.Image) -> float:
    """Compute LPIPS perceptual distance (lower is better)."""
    model = _load_lpips()
    device = next(model.parameters()).device

    # Convert to tensor: (1, 3, H, W) in range [-1, 1]
    def img_to_tensor(img):
        img = img.convert("RGB").resize((1024, 1024))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2 - 1  # Scale to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    canvas_tensor = img_to_tensor(canvas)
    target_tensor = img_to_tensor(target)

    with torch.no_grad():
        distance = model(canvas_tensor, target_tensor).item()

    return float(distance)


def evaluate(
    canvas: Image.Image,
    target: Image.Image | None,
    compute_all: bool = True,
    benchmark_type: str = "target_image",
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        canvas: Final canvas image
        target: Target image
        compute_all: If False, skip expensive metrics (CLIP, LPIPS)

    Returns:
        Dict with metrics: mse, ssim, clip_similarity, lpips
    """
    metrics = {}

    if benchmark_type == "text_prompt" or target is None:
        print("Skipping evaluation for text prompt benchmark, not implemented yet")
        return metrics

    # Always compute basic metrics
    metrics["mse"] = compute_mse(canvas, target)
    metrics["ssim"] = compute_ssim(canvas, target)

    if compute_all:
        try:
            metrics["clip_similarity"] = compute_clip_similarity(canvas, target)
        except Exception as e:
            print(f"Warning: Could not compute CLIP similarity: {e}")
            metrics["clip_similarity"] = None

        try:
            metrics["lpips"] = compute_lpips(canvas, target)
        except Exception as e:
            print(f"Warning: Could not compute LPIPS: {e}")
            metrics["lpips"] = None

    return metrics
