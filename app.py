"""
CIFAR-10 Interactive Classification — Streamlit Demo
=====================================================

A web-based demo app for classifying images using the trained Custom CNN
and MobileNetV2 models. Upload any image or sample from CIFAR-10 test set.

Run:
    streamlit run app.py

Author:  Pouya Alavi
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Streamlit is required: pip install streamlit"
    )


# ============================================================================
#  Constants
# ============================================================================

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

class CustomCNN(nn.Module):
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
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


# ============================================================================
#  Model Loading
# ============================================================================

def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_models():
    """Load both models once and cache them across Streamlit reruns."""
    import os
    device = _select_device()
    loaded = {}

    search_dirs = ["checkpoints", "models", "."]

    # Custom CNN
    model_cnn = CustomCNN(num_classes=10)
    cnn_candidates = [
        "custom_cnn_best.pth", "custom_cnn_final.pth",
        "custom_cnn_model.pth", "custom_cnn.pth",
    ]
    for d in search_dirs:
        for c in cnn_candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                state = torch.load(p, map_location=device, weights_only=True)
                model_cnn.load_state_dict(state)
                model_cnn.to(device)
                model_cnn.eval()
                loaded["Custom CNN"] = model_cnn
                break
        if "Custom CNN" in loaded:
            break

    # MobileNetV2
    model_mn = build_mobilenetv2(num_classes=10)
    mn_candidates = [
        "mobilenetv2_best.pth", "mobilenetv2_final.pth",
        "mobilenet_model.pth", "mobilenet.pth",
    ]
    for d in search_dirs:
        for c in mn_candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                state = torch.load(p, map_location=device, weights_only=True)
                model_mn.load_state_dict(state)
                model_mn.to(device)
                model_mn.eval()
                loaded["MobileNetV2"] = model_mn
                break
        if "MobileNetV2" in loaded:
            break

    return loaded, device


# ============================================================================
#  Prediction
# ============================================================================

def predict(model, image: Image.Image, model_name: str, device, top_k: int = 5):
    """Run inference and return top-k predictions."""
    if model_name == "Custom CNN":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)
    return [(CLASS_NAMES[i.item()], p.item() * 100) for p, i in zip(top_probs, top_indices)]


# ============================================================================
#  Streamlit UI
# ============================================================================

def main():
    st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")

    st.title("CIFAR-10 Image Classification")
    st.markdown(
        "Compare a **Custom CNN** (trained from scratch) against **MobileNetV2** "
        "(transfer learning) on image classification."
    )

    # Load models
    with st.spinner("Loading models..."):
        loaded_models, device = load_models()

    if not loaded_models:
        st.error(
            "No model checkpoints found. Run the notebook first to generate "
            "model weights in the checkpoints/ directory."
        )
        return

    st.sidebar.header("Settings")
    available_models = list(loaded_models.keys())
    selected_models = st.sidebar.multiselect(
        "Models to compare", available_models, default=available_models,
    )
    top_k = st.sidebar.slider("Top-K predictions", 1, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Device:** `{device}`")
    st.sidebar.markdown(f"**Models loaded:** {len(loaded_models)}")

    # Input selection
    tab_upload, tab_cifar = st.tabs(["Upload Image", "CIFAR-10 Test Sample"])

    image = None
    true_label = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

    with tab_cifar:
        col1, col2 = st.columns([1, 3])
        with col1:
            sample_idx = st.number_input("Image index (0-9999)", 0, 9999, 42)
            if st.button("Load sample"):
                dataset = torchvision.datasets.CIFAR10(
                    root="./data", train=False, download=True,
                    transform=transforms.ToTensor(),
                )
                tensor_img, label = dataset[sample_idx]
                image = transforms.ToPILImage()(tensor_img)
                true_label = CLASS_NAMES[label]
        with col2:
            if st.button("Random sample"):
                dataset = torchvision.datasets.CIFAR10(
                    root="./data", train=False, download=True,
                    transform=transforms.ToTensor(),
                )
                idx = np.random.randint(len(dataset))
                tensor_img, label = dataset[idx]
                image = transforms.ToPILImage()(tensor_img)
                true_label = CLASS_NAMES[label]
                st.info(f"Loaded random sample (index {idx})")

    # Display and classify
    if image is not None:
        cols = st.columns(1 + len(selected_models))

        with cols[0]:
            st.image(image, caption="Input Image", use_container_width=True)
            if true_label:
                st.markdown(f"**True label:** {true_label}")

        for i, model_name in enumerate(selected_models):
            model = loaded_models[model_name]
            preds = predict(model, image, model_name, device, top_k=top_k)

            with cols[i + 1]:
                st.subheader(model_name)
                for cls, conf in preds:
                    st.progress(conf / 100, text=f"{cls}: {conf:.1f}%")


if __name__ == "__main__":
    main()
