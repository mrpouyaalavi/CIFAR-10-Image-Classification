"""
Grad-CAM — Gradient-weighted Class Activation Mapping
=====================================================

Generates visual explanations for CNN predictions by highlighting the image
regions most influential to a given classification decision. This is an
implementation of the technique described in:

    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Why Grad-CAM matters:
    Deep CNNs are often treated as black boxes. Grad-CAM provides
    interpretability by computing a coarse localisation map that shows
    *where* the model is looking when it makes a prediction. This is
    valuable for debugging misclassifications, building trust in model
    decisions, and verifying that the network learns meaningful features
    rather than exploiting dataset shortcuts (e.g., background colour).

How it works (high-level):
    1. Perform a forward pass and record the feature maps at a target
       convolutional layer.
    2. Backpropagate the score of the target class to compute gradients
       with respect to those feature maps.
    3. Global-average-pool the gradients to obtain per-channel importance
       weights — channels that strongly influence the target class get
       higher weights.
    4. Compute a weighted sum of the feature maps and apply ReLU (we only
       care about features that positively influence the target class).
    5. The resulting heatmap is resized to the input image dimensions and
       overlaid as a colour map.

Usage Examples:
    python gradcam.py --model custom_cnn --image-index 0
    python gradcam.py --model mobilenet --image-index 42 --save results/gradcam/
    python gradcam.py --model both --num-images 8 --save results/gradcam/

Author:  Pouya Alavi
License: MIT
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================================
#  Constants — Dataset Statistics & Class Labels
# ============================================================================
# See predict.py for detailed explanations of these normalisation values.

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
#  Model Definitions
# ============================================================================
# Architecture definitions must exactly match those used during training.
# See predict.py for detailed architectural documentation.
# ============================================================================

class CustomCNN(nn.Module):
    """4-block CNN matching the notebook architecture. See predict.py for details."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — Low-level features (edges, colour gradients)
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2 — Mid-level features (textures, patterns)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3 — High-level features (object parts)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 4 — Semantic features (whole-object representations)
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
    """Build MobileNetV2 with frozen backbone. See predict.py for details."""
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


# ============================================================================
#  Grad-CAM Implementation
# ============================================================================
# This is the core of the script — a reusable Grad-CAM class that can be
# attached to any convolutional layer in any PyTorch model.
#
# Key concept: PyTorch hooks
#   - A forward hook captures the feature map activations at a target layer
#     during the forward pass (without modifying the computation graph).
#   - A backward hook captures the gradients flowing back through that layer
#     during backpropagation.
#   These two captured tensors are combined to produce the Grad-CAM heatmap.
# ============================================================================

class GradCAM:
    """
    Grad-CAM: generates a class-discriminative localisation map for any
    target convolutional layer in a classification network.

    The algorithm highlights image regions that most strongly activate the
    target class. Unlike occlusion-based methods (which require many forward
    passes), Grad-CAM needs only one forward + one backward pass, making it
    computationally efficient.

    Parameters:
        model:        The classification model (must be in inference mode).
        target_layer: The convolutional layer whose activations and gradients
                      are used to compute the heatmap. Typically the last conv
                      layer is chosen because it has the richest semantic
                      features while still retaining spatial information.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Register PyTorch hooks to intercept activations and gradients
        # at the target layer without modifying the model's forward() method.
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ───────────────────────────────────────────

    def _save_activation(self, module, input, output):
        """Forward hook: capture the target layer's output feature maps."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: capture gradients flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    # ── Main API ─────────────────────────────────────────────────

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single input image.

        Algorithm steps:
            1. Forward pass — produces logits and triggers the forward hook
               to capture feature maps A at the target layer.
            2. Select the target class score y_c (predicted class if not
               specified) and backpropagate to get gradients dA/dy_c.
            3. Global-average-pool the gradients across spatial dimensions
               to obtain per-channel importance weights alpha_c.
            4. Compute the weighted sum: sum(alpha_c * A_c) across channels.
            5. Apply ReLU — only keep positive contributions (features that
               increase the target class score, not decrease it).
            6. Normalise to [0, 1] for visualisation.

        Parameters:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Class index to explain. If None, the model's
                          predicted (argmax) class is used.

        Returns:
            heatmap:      2D numpy array in [0, 1], sized to the target
                          layer's spatial output dimensions.
            target_class: The class index that was explained.
            output:       Raw model logits for downstream use.
        """
        self.model.eval()
        self.model.zero_grad()

        # Step 1: Forward pass — activations are captured by the hook
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Step 2: Backpropagate the target class score
        score = output[0, target_class]
        score.backward()

        # Step 3: Channel importance weights via global average pooling
        # Shape: (1, C, H', W') -> (1, C, 1, 1)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # Step 4: Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)

        # Step 5: ReLU — discard negative influence (regions that suppress
        # the target class are not useful for localisation)
        cam = F.relu(cam)

        # Step 6: Normalise to [0, 1] for heatmap visualisation
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, output

    def remove_hooks(self):
        """Detach hooks to prevent memory leaks when done with Grad-CAM."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ============================================================================
#  Visualisation Helpers
# ============================================================================

def _denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """
    Reverse channel-wise normalisation and convert to a displayable image.

    During preprocessing, images are normalised as:
        pixel_norm = (pixel - mean) / std

    To recover the original pixel values for display:
        pixel_orig = pixel_norm * std + mean

    Parameters:
        tensor: Normalised image tensor of shape (C, H, W).
        mean:   Per-channel mean used during normalisation.
        std:    Per-channel standard deviation used during normalisation.

    Returns:
        A uint8 numpy array of shape (H, W, C) suitable for plt.imshow().
    """
    img = tensor.clone().cpu()
    for c, m, s in zip(img, mean, std):
        c.mul_(s).add_(m)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original image using a jet colourmap.

    The heatmap is first resized to match the original image dimensions using
    bilinear interpolation, then blended with the image at the specified
    opacity (alpha). The jet colourmap maps low activations to blue and high
    activations to red, making it easy to spot the most important regions.

    Parameters:
        image:   Original image as a uint8 array of shape (H, W, 3).
        heatmap: Grad-CAM heatmap as a float array in [0, 1].
        alpha:   Blending factor — 0.0 = original image, 1.0 = heatmap only.

    Returns:
        overlay:         Blended image as a uint8 array.
        heatmap_resized: The resized heatmap (useful for standalone display).
    """
    from PIL import Image as PILImage

    h, w = image.shape[:2]

    # Resize heatmap from the target layer's spatial size to the image size
    heatmap_resized = np.array(
        PILImage.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize((w, h), resample=PILImage.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply jet colourmap: blue (low activation) -> red (high activation)
    colormap = cm.jet(heatmap_resized)[..., :3]
    colormap = (colormap * 255).astype(np.uint8)

    # Alpha-blend the colourmap with the original image
    overlay = (alpha * colormap + (1 - alpha) * image).astype(np.uint8)
    return overlay, heatmap_resized


# ============================================================================
#  Grid Visualisation
# ============================================================================

def visualize_gradcam_grid(
    model: nn.Module,
    target_layer: nn.Module,
    dataset,
    indices: list[int],
    mean: tuple,
    std: tuple,
    model_name: str = "Model",
    save_path: str | None = None,
):
    """
    Create a grid of Grad-CAM visualisations for multiple test images.

    Layout — each row displays three panels:
        [Original Image]  [Grad-CAM Heatmap]  [Overlay]

    The title of each original image is colour-coded:
        - Green: the model's prediction matches the ground truth.
        - Red:   the model misclassified the image.

    This side-by-side layout makes it easy to compare what the model
    "sees" (heatmap) against what a human sees (original image), and to
    verify that the model attends to semantically meaningful regions
    (e.g., the animal's body, not the background).

    Parameters:
        model:        Trained model in inference mode.
        target_layer: Convolutional layer to generate heatmaps from.
        dataset:      CIFAR-10 test dataset with the appropriate transforms.
        indices:      List of dataset indices to visualise.
        mean:         Normalisation mean (for denormalisation).
        std:          Normalisation std (for denormalisation).
        model_name:   Display name for the figure title.
        save_path:    Optional path to save the figure.
    """
    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer)
    n = len(indices)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=16, fontweight="bold", y=1.01)

    for row, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Generate the Grad-CAM heatmap
        heatmap, pred_class, logits = gradcam(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        confidence = probs[pred_class].item() * 100

        # Recover the original image for display
        orig_img = _denormalize(img_tensor, mean, std)
        overlay, heatmap_resized = overlay_heatmap(orig_img, heatmap)

        # Column 0 — Original image with prediction label
        axes[row, 0].imshow(orig_img)
        colour = "green" if pred_class == true_label else "red"
        axes[row, 0].set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_class]} ({confidence:.1f}%)",
            fontsize=9,
            color=colour,
        )
        axes[row, 0].axis("off")

        # Column 1 — Raw Grad-CAM heatmap (jet colourmap)
        axes[row, 1].imshow(heatmap_resized, cmap="jet")
        axes[row, 1].set_title("Grad-CAM heatmap", fontsize=9)
        axes[row, 1].axis("off")

        # Column 2 — Overlay (heatmap blended with original)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

    # Clean up hooks to prevent memory leaks in long-running sessions
    gradcam.remove_hooks()


# ============================================================================
#  Utilities — Device Selection & Model Loading
# ============================================================================

def _select_device():
    """Auto-detect the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _remap_legacy_keys(state_dict: dict) -> dict:
    """
    Remap state dict keys from legacy conv_block format to features.* format.

    See predict.py for full documentation on the key-mapping logic.
    """
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


def _load_model(name: str, device: torch.device):
    """Load a saved model from the standard checkpoint paths."""
    search_dirs = ["checkpoints", "models", "."]
    if name == "custom_cnn":
        model = CustomCNN(num_classes=10)
        candidates = [
            "custom_cnn_best.pth", "custom_cnn_final.pth",
            "custom_cnn_model.pth", "custom_cnn.pth",
            "custom_cnn_best_cpufast.pth",
        ]
    elif name == "mobilenet":
        model = build_mobilenetv2(num_classes=10)
        candidates = [
            "mobilenetv2_best.pth", "mobilenetv2_final.pth",
            "mobilenet_model.pth", "mobilenet.pth",
        ]
    else:
        raise ValueError(f"Unknown model: {name}")

    for d in search_dirs:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                state = torch.load(p, map_location=device, weights_only=True)
                if name == "custom_cnn":
                    state = _remap_legacy_keys(state)
                try:
                    model.load_state_dict(state)
                except RuntimeError:
                    continue
                model.to(device).eval()
                print(f"Loaded {name} from {p}")
                return model
    raise FileNotFoundError(f"No checkpoint found for {name}")


def _get_target_layer(model, name: str):
    """
    Return the target convolutional layer for Grad-CAM heatmap generation.

    Why the last conv layer?
        Earlier layers capture low-level patterns (edges, textures) that
        are too generic for class-specific localisation. The last conv layer
        captures the highest-level semantic features while still retaining
        spatial structure — making it ideal for Grad-CAM.

    For Custom CNN:
        Block 3's last Conv2d (features[22], the 256->256 conv) which
        outputs an 8x8 feature map — a good balance between spatial
        resolution and semantic richness. Block 4's Conv2d only produces
        a 4x4 map (just 16 positions), which yields overly coarse heatmaps.

    For MobileNetV2:
        The final inverted residual block (model.features[-1]), which outputs
        1280-channel feature maps at 7x7 spatial resolution.
    """
    if name == "custom_cnn":
        # Block 3's last Conv2d at index 19 (256->256) outputs 8x8 spatial
        # resolution, providing 64 meaningful positions for localisation.
        # (Block 4's Conv2d at index 24 only outputs 4x4 — too coarse.)
        return model.features[19]
    elif name == "mobilenet":
        return model.features[-1]
    raise RuntimeError("Could not find target layer")


# ============================================================================
#  CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for CIFAR-10 models")
    parser.add_argument(
        "--model", choices=["custom_cnn", "mobilenet", "both"], default="both",
        help="Which model to visualize",
    )
    parser.add_argument("--num-images", type=int, default=6, help="Number of images to visualize")
    parser.add_argument("--image-index", type=int, nargs="*", help="Specific image indices")
    parser.add_argument("--save", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    args = parser.parse_args()

    device = _select_device()
    print(f"Device: {device}")

    # ── Prepare CIFAR-10 test sets with model-specific transforms ──
    # Each model needs its own dataset instance because preprocessing
    # differs: Custom CNN uses 32x32 with CIFAR stats, MobileNetV2
    # uses 224x224 with ImageNet stats.

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_mobilenet = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    testset_cifar = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_cifar,
    )
    testset_mobilenet = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_mobilenet,
    )

    # ── Select image indices ─────────────────────────────────────
    if args.image_index:
        indices = args.image_index
    else:
        np.random.seed(args.seed)
        indices = np.random.choice(len(testset_cifar), size=args.num_images, replace=False).tolist()
    print(f"Image indices: {indices}")

    # ── Generate Grad-CAM visualisations per model ───────────────
    models_to_run = (
        ["custom_cnn", "mobilenet"] if args.model == "both" else [args.model]
    )

    for model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"  Grad-CAM — {model_name}")
        print(f"{'='*50}")

        model = _load_model(model_name, device)
        target_layer = _get_target_layer(model, model_name)

        # Select the correct dataset and normalisation stats
        dataset = testset_cifar if model_name == "custom_cnn" else testset_mobilenet
        mean = CIFAR_MEAN if model_name == "custom_cnn" else IMAGENET_MEAN
        std = CIFAR_STD if model_name == "custom_cnn" else IMAGENET_STD

        save_path = None
        if args.save:
            save_path = os.path.join(args.save, f"gradcam_{model_name}.png")

        visualize_gradcam_grid(
            model=model,
            target_layer=target_layer,
            dataset=dataset,
            indices=indices,
            mean=mean,
            std=std,
            model_name=model_name.replace("_", " ").title(),
            save_path=save_path,
        )

    print("\n✅ Grad-CAM visualization complete!")


if __name__ == "__main__":
    main()
