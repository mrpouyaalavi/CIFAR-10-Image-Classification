"""
Shared model definitions, loading utilities, and Grad-CAM for CIFAR-10.

This module is the single source of truth for model architectures and
checkpoint loading. It is imported by app.py (Gradio demo on Hugging Face
Spaces), and can also be used by predict.py and gradcam.py directly.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================================
#  Constants
# ============================================================================

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Map internal CLI model keys → pretty display names used in every figure
# title, log line, and chart heading. This exists because a naive
# `.replace("_"," ").title()` mangles proper nouns:
#     "custom_cnn"  →  "Custom Cnn"      (should be "Custom CNN")
#     "mobilenet"   →  "Mobilenet"       (should be "MobileNetV2")
# Consolidating here prevents the display-name drift we caught in the
# CLI smoke-test screenshots.
PRETTY_NAMES: dict[str, str] = {
    "custom_cnn": "Custom CNN",
    "mobilenet":  "MobileNetV2",
    # Already-pretty keys map to themselves so callers can safely pass either
    "Custom CNN":      "Custom CNN",
    "MobileNetV2":     "MobileNetV2",
    "ResNet-18":       "ResNet-18",
    "EfficientNet-B0": "EfficientNet-B0",
    "ViT-Small":       "ViT-Small",
}


def pretty_model_name(name: str) -> str:
    """Return the canonical display name for a model key.

    Accepts either the CLI snake-case key ("custom_cnn") or the already-pretty
    name ("Custom CNN"). Unknown names fall back to the input unchanged so new
    architectures don't silently crash the CLI.
    """
    return PRETTY_NAMES.get(name, name)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
#  Model Architectures
# ============================================================================

class CustomCNN(nn.Module):
    """4-block CNN trained from scratch on CIFAR-10 (3 -> 64 -> 128 -> 256 -> 512)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_mobilenetv2(num_classes: int = 10) -> nn.Module:
    """MobileNetV2 with frozen ImageNet backbone and trainable classifier head."""
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 with frozen ImageNet backbone and trainable FC head.

    Architecture: standard torchvision ResNet-18 (11.2 M total params) with
    every backbone parameter frozen. Only the final fully-connected layer
    (512 × 10 + 10 = 5 130 trainable params) is replaced and trained.
    """
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 10) -> nn.Module:
    """EfficientNet-B0 with frozen ImageNet backbone and trainable classifier head.

    Architecture: standard torchvision EfficientNet-B0 (4.0 M total params)
    with every backbone parameter frozen. Only the classifier head
    (Dropout + Linear 1280 × 10 = 12 810 trainable params) is replaced.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


# ============================================================================
#  Device Selection
# ============================================================================
#
# Single source of truth for device auto-detection across every surface in
# this repo (training, evaluation, CLI inference, Grad-CAM, Gradio demo).
# The goal: pick the best accelerator available, and print it exactly once
# per process so there's always a clear record in the logs.
#
# Priority order:
#   1. CUDA   — NVIDIA GPU (servers, Colab, Kaggle, most cloud instances)
#   2. MPS    — Apple Silicon GPU (M1/M2/M3/M4 Macs)
#   3. CPU    — universal fallback
#
# The `_device_logged` module-level flag ensures we only print once even
# when multiple callers (Gradio reruns, test suites, notebooks)
# hit select_device() repeatedly.

_device_logged: bool = False


def describe_device(device: torch.device) -> str:
    """Return a human-readable description of the selected device + hardware."""
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"CUDA — {name} ({mem_gb:.1f} GB)"
    if device.type == "mps":
        import platform
        # There's no public torch API for MPS device name; use the OS machine
        # identifier as a proxy (e.g. "arm64" on Apple Silicon).
        return f"MPS — Apple Silicon ({platform.machine()})"
    # CPU — try to surface brand string when possible
    import platform
    proc = platform.processor() or platform.machine() or "unknown"
    return f"CPU — {proc}"


def select_device(verbose: bool = True) -> torch.device:
    """
    Auto-select the best available PyTorch device (CUDA > MPS > CPU).

    Parameters
    ----------
    verbose : bool, default True
        If True *and* this is the first call in the current process, print a
        one-line summary of the selected device to stdout. Subsequent calls
        stay silent regardless of `verbose` so we don't spam the console on
        every Gradio rerun. Pass `verbose=False` to silence the first call
        too (useful for tests).

    Returns
    -------
    torch.device
        Ready-to-use device handle. Pass it to `.to(device)` or
        `torch.load(..., map_location=device)`.
    """
    global _device_logged

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose and not _device_logged:
        print(f"[cifar10] Device: {describe_device(device)}", flush=True)
        _device_logged = True

    return device


# ============================================================================
#  Checkpoint Loading
# ============================================================================

def _remap_legacy_keys(state_dict: dict) -> dict:
    """Remap legacy conv_block* keys to features.* format."""
    if any(k.startswith("features.") for k in state_dict):
        return state_dict
    if not any(k.startswith("conv_block") for k in state_dict):
        return state_dict

    block_offsets = {"conv_block1": 0, "conv_block2": 8, "conv_block3": 16, "conv_block4": 24}
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for block_name, offset in block_offsets.items():
            if key.startswith(block_name + "."):
                suffix = key[len(block_name) + 1:]
                layer_idx = int(suffix.split(".")[0])
                rest = ".".join(suffix.split(".")[1:])
                new_key = f"features.{offset + layer_idx}.{rest}"
                break
        remapped[new_key] = value
    return remapped


SEARCH_DIRS = ["checkpoints", "models", "."]

CNN_CANDIDATES = [
    "custom_cnn_best.pth", "custom_cnn_final.pth",
    "custom_cnn_model.pth", "custom_cnn.pth",
    "custom_cnn_best_cpufast.pth",
]

MN_CANDIDATES = [
    "mobilenetv2_best.pth", "mobilenetv2_final.pth",
    "mobilenet_model.pth", "mobilenet.pth",
]


# ── Hugging Face Hub integration ─────────────────────────────────────────
#
# Model weights are stored in a separate HF model repository to keep the
# Space repo lightweight (code only, no binaries).  On first boot, weights
# are downloaded from the Hub and cached.  Subsequent boots use the cache.
#
# Repository: mrpouyaalavi/cifar10-models
# Files:      custom_cnn_best.pth, mobilenetv2_best.pth, resnet-18_best.pth

HF_MODEL_REPO = "mrpouyaalavi/cifar10-models"

# Maps each checkpoint filename to its filename in the HF model repo.
HF_CHECKPOINT_FILES: dict[str, str] = {
    "custom_cnn_best.pth": "custom_cnn_best.pth",
    "mobilenetv2_best.pth": "mobilenetv2_best.pth",
    "resnet-18_best.pth": "resnet-18_best.pth",
}


def _download_from_hub(filename: str) -> str | None:
    """Download a checkpoint from the HF model repo.  Returns local path or None."""
    if filename not in HF_CHECKPOINT_FILES:
        return None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_CHECKPOINT_FILES[filename],
        )
        return path
    except Exception as exc:
        print(f"[cifar10] Hub download failed for {filename}: {exc}", flush=True)
        return None


# ── Model registry ─────────────────────────────────────────────────────────
#
# Single source of truth for "which architectures this app can load, and how".
# Each entry has:
#   builder      - zero-arg callable that returns a fresh nn.Module instance
#   candidates   - ordered list of checkpoint filenames to probe on disk
#   needs_remap  - whether to run _remap_legacy_keys() on the state dict
#   hub_file     - primary checkpoint to download from HF Hub if not local

MODEL_REGISTRY: dict[str, dict] = {
    "Custom CNN": {
        "builder": lambda: CustomCNN(num_classes=10),
        "candidates": CNN_CANDIDATES,
        "needs_remap": True,
        "hub_file": "custom_cnn_best.pth",
    },
    "MobileNetV2": {
        "builder": lambda: build_mobilenetv2(num_classes=10),
        "candidates": MN_CANDIDATES,
        "needs_remap": False,
        "hub_file": "mobilenetv2_best.pth",
    },
    "ResNet-18": {
        "builder": lambda: build_resnet18(num_classes=10),
        "candidates": ["resnet-18_best.pth", "resnet18_best.pth"],
        "needs_remap": False,
        "hub_file": "resnet-18_best.pth",
    },
    "EfficientNet-B0": {
        "builder": lambda: build_efficientnet_b0(num_classes=10),
        "candidates": ["efficientnet-b0_best.pth", "efficientnet_b0_best.pth"],
        "needs_remap": False,
        "hub_file": "efficientnet-b0_best.pth",
    },
}


def _find_checkpoint(candidates: list[str]) -> str | None:
    """Return the first candidate checkpoint path that exists on disk."""
    for d in SEARCH_DIRS:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                return p
    return None


def _resolve_checkpoint(info: dict) -> str | None:
    """Find a checkpoint locally, or download from HF Hub as fallback."""
    path = _find_checkpoint(info["candidates"])
    if path is not None:
        return path
    hub_file = info.get("hub_file")
    if hub_file:
        return _download_from_hub(hub_file)
    return None


def list_available_models() -> list[str]:
    """Return display names of models whose checkpoint can be resolved.

    Checks local disk first, then HF Hub.  On HF Spaces the Hub download
    is essentially instant (same data centre), so this stays fast.
    """
    return [
        name
        for name, info in MODEL_REGISTRY.items()
        if _resolve_checkpoint(info) is not None
    ]


def load_model_by_name(name: str, device: torch.device) -> nn.Module:
    """Instantiate, load weights, and return a model in inference mode.

    Checkpoints are resolved in order:
      1. Local filesystem  (checkpoints/, models/, ./)
      2. Hugging Face Hub  (cached after first fetch)
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'.  Registered: {list(MODEL_REGISTRY)}")

    info = MODEL_REGISTRY[name]
    path = _resolve_checkpoint(info)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint found for {name} (local dirs: {SEARCH_DIRS}, "
            f"Hub repo: {HF_MODEL_REPO}).  Tried: {info['candidates']}"
        )

    model = info["builder"]()
    state = torch.load(path, map_location=device, weights_only=True)
    if info["needs_remap"]:
        state = _remap_legacy_keys(state)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_models(device: torch.device) -> dict[str, nn.Module]:
    """Load every available model at once.  Kept for backward compatibility."""
    return {
        name: load_model_by_name(name, device)
        for name in list_available_models()
    }


# ============================================================================
#  Preprocessing
# ============================================================================

def get_transform(model_name: str):
    """Return the preprocessing pipeline for the given model."""
    import torchvision.transforms as transforms

    if model_name == "Custom CNN":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ============================================================================
#  Inference
# ============================================================================

def predict(
    model: nn.Module,
    image: Image.Image,
    model_name: str,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Run inference and return top-k (class_name, confidence_pct) tuples."""
    transform = get_transform(model_name)
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    # torch.inference_mode() is strictly faster than torch.no_grad() because
    # it also disables version counters and view tracking. Safe here because
    # we never need autograd on the returned logits — Grad-CAM runs its own
    # forward pass with requires_grad=True on the input tensor.
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)
    return [(CLASS_NAMES[i.item()], p.item() * 100) for p, i in zip(top_probs, top_indices)]


# ============================================================================
#  Grad-CAM
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping for visual explanations."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations = None
        self._gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """
        Return (heatmap_0_to_1, predicted_or_target_class, logits).

        The logits are returned detached so callers can compute confidence
        scores without re-running the forward pass. This avoids the common
        mistake of calling the model twice for Grad-CAM + confidence.
        """
        self.model.train(False)
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, output.detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the best convolutional-feature layer for Grad-CAM heatmaps.

    We target the *last* high-resolution conv feature map (just before global
    pooling) because that's where the network has its richest spatial +
    semantic representation — early layers see edges, late layers see objects.

    Important: we must NOT hook a layer whose output is subsequently modified
    by an in-place op (e.g. ReLU(inplace=True)), because the backward hook
    triggers an autograd error when it sees the in-place modification. For
    the Custom CNN the last Conv2d at features[24] is the safest choice.
    """
    if model_name == "Custom CNN":
        # features[24] = last Conv2d(256→512). Its output is read (not modified)
        # by the subsequent BatchNorm, so the backward hook stays consistent.
        return model.features[24]

    # MobileNetV2: features[-1] is a Conv2dNormActivation container
    # (Conv + BN + ReLU6-inplace). Hooking the container triggers the
    # "NoneType gradients" bug because backward hooks on containers don't
    # always fire; hook the inner Conv2d directly for reliable activations.
    if model_name == "MobileNetV2":
        last_block = model.features[-1]
        # Conv2dNormActivation exposes the conv as [0]
        return last_block[0] if hasattr(last_block, "__getitem__") else last_block

    # ResNet-18: layer4[-1].conv2 is the last Conv2d (512 channels, 7×7
    # spatial for 224×224 input). It sits before the BasicBlock's residual
    # addition, so the backward hook fires cleanly.
    if model_name == "ResNet-18":
        return model.layer4[-1].conv2

    # EfficientNet-B0: same Conv2dNormActivation pattern as MobileNetV2.
    if model_name == "EfficientNet-B0":
        last_block = model.features[-1]
        return last_block[0] if hasattr(last_block, "__getitem__") else last_block

    # Fallback for any other architecture
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError(f"No known Grad-CAM target layer for {model_name}")


def compute_gradcam_overlay(
    model: nn.Module,
    image: Image.Image,
    model_name: str,
    device: torch.device,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Compute Grad-CAM for an uploaded image.

    Returns:
        overlay:    Blended image (H, W, 3) uint8
        heatmap:    Raw heatmap resized to image dimensions (H, W) float
        pred_class: Predicted class index
        confidence: Confidence percentage
    """
    import matplotlib.cm as mpl_cm

    target_layer = get_gradcam_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    # Preprocess exactly once. We set requires_grad=True on the *input tensor*
    # so that autograd builds a full backward graph through every intermediate
    # layer even when the model's parameters are frozen (e.g. MobileNetV2's
    # transfer-learning backbone). Without this, backward hooks on frozen
    # layers never fire and self._gradients stays None.
    transform = get_transform(model_name)
    tensor = transform(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    try:
        heatmap, pred_class, logits = gradcam.generate(tensor, target_class=None)
    finally:
        gradcam.remove_hooks()

    with torch.no_grad():
        probs = F.softmax(logits, dim=1)[0]
        confidence = probs[pred_class].item() * 100

    # Resize heatmap to original image size and create overlay
    orig_np = np.array(image.convert("RGB"))
    h, w = orig_np.shape[:2]

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), resample=Image.Resampling.BILINEAR
        )
    ).astype(np.float32) / 255.0

    colormap = mpl_cm.jet(heatmap_resized)[..., :3]
    colormap = (colormap * 255).astype(np.uint8)

    overlay = (alpha * colormap + (1 - alpha) * orig_np).astype(np.uint8)

    return overlay, heatmap_resized, pred_class, confidence
