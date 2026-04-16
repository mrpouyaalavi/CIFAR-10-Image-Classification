---
title: CIFAR-10 Image Classification
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
---

<div align="center">

<!-- Typing animation -->
[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2800&pause=700&color=EE4C2C&center=true&vCenter=true&width=860&lines=CIFAR-10+Deep+Learning+Image+Classification;Custom+CNN+vs+Transfer+Learning+%E2%80%94+A+Comparative+Study;PyTorch+%C2%B7+Grad-CAM+%C2%B7+MobileNetV2+%C2%B7+Gradio;5+Architectures+%C2%B7+86.91%25+Accuracy+%C2%B7+Full+ML+Pipeline)](https://readme-typing-svg.demolab.com)

<!-- Badges -->
![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Demo_App-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD21E?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

# CIFAR-10 Image Classification — From Deep Learning Training to Deployment

> **An end-to-end deep learning project that designs, trains, and evaluates multiple architectures on the CIFAR-10 benchmark, demonstrating the efficiency of transfer learning compared with a custom CNN baseline.**

This project is a comprehensive, portfolio-grade machine learning study that goes beyond model training. It includes data augmentation pipelines (RandomCrop, CutOut, MixUp, CutMix), cosine annealing learning rate scheduling, progressive unfreezing, INT8 model quantisation, Grad-CAM interpretability visualisations, a CLI inference toolkit, and a Gradio demo app deployed on Hugging Face Spaces — all documented in a structured Jupyter Notebook.

**[📓 Explore the Notebook](cifar10%20image%20classification.ipynb)** &nbsp;·&nbsp; **[🚀 Live Demo (Hugging Face)](https://huggingface.co/spaces/mrpouyaalavi/CIFAR-10-Image-Classification)** &nbsp;·&nbsp; **[📊 Key Results](#-key-results--performance-benchmarks)**

> **Note:** The original Streamlit demo at [cifar10-pouyaalavi.streamlit.app](https://cifar10-pouyaalavi.streamlit.app/) now serves as a landing page that redirects to the new Hugging Face Space.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🎯 Motivation & Research Question

> _How much does a pretrained backbone actually help compared to training from scratch — when both models share the same training budget?_

Deep learning practitioners often default to transfer learning without quantifying its advantage under controlled conditions. This project answers that question through a **rigorous, controlled experiment**: identical dataset, optimiser, learning rate, epoch count, and augmentation strategy — the only variable is the architecture and whether the weights are pretrained.

The results have direct implications for:

- **Model selection** in resource-constrained environments (edge, mobile, embedded)
- **Training efficiency** when labelled data is limited
- **Deployment strategy** when balancing latency versus accuracy

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📊 Key Results & Performance Benchmarks

<div align="center">

| Metric | Custom CNN | MobileNetV2 | ResNet-18 | Winner |
|:-------|:----------:|:-----------:|:---------:|:------:|
| **Test Accuracy** | 48.40% | **86.91%** | 82.10% | 🏆 MobileNetV2 |
| **Trainable Params** | 2,462,282 | 12,810 | **5,130** | 🏆 ResNet-18 |
| **Model Size** | **9.42 MB** | 8.76 MB | 42.73 MB | 🏆 MobileNetV2 |
| **CPU Latency (batch 1)** | **1.38 ms** | 17.22 ms | 9.80 ms | 🏆 Custom CNN |
| **Throughput** | **~724 FPS** | ~58 FPS | ~102 FPS | 🏆 Custom CNN |

</div>

> All numbers are measured empirically from the committed checkpoints on the full 10 000-image CIFAR-10 test set (accuracy verified 2026-04-10; Custom CNN latency re-measured 2026-04-11 with 100 trials on Apple silicon CPU, batch 1). No cherry-picking, no leftover notebook metrics.

> **Key Finding:** MobileNetV2 achieves **86.91% accuracy** with just **0.5%** of the Custom CNN's trainable parameters — a **+38.5 percentage-point** lift for a **192× reduction in trainable weights**. Transfer learning doesn't just win on accuracy; it wins while training almost nothing at all.

### Training Progression — Convergence Comparison

```text
Epoch   Custom CNN (Val Acc)     MobileNetV2 (Val Acc)
─────   ────────────────────     ─────────────────────
  1          21.0%                    85.88%  ◀ Already ~86% after ONE epoch
  2          27.8%                    86.80%
  3          32.4%                    86.91%  ◀ Effectively converged
  …            …                        …
 15          48.40%                   86.91%  ◀ Final
```

MobileNetV2 reaches **85.88% after a single epoch** and plateaus within three. The Custom CNN is still climbing all 15 epochs — a direct demonstration of the value of pretrained feature representations, even when the input resolutions differ drastically (ImageNet 224×224 vs. CIFAR-10 32×32).

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🧠 Model Architectures

### Custom CNN — 4-Block Design (Trained From Scratch)

A purpose-built convolutional network with progressive channel expansion, dual convolutions per block, and aggressive regularisation:

```text
Input (3 × 32 × 32)
  │
  ├── Block 1: Conv(3→64) ×2 → BatchNorm → ReLU → MaxPool → Dropout(0.25)     [32→16]
  ├── Block 2: Conv(64→128) ×2 → BatchNorm → ReLU → MaxPool → Dropout(0.25)   [16→8]
  ├── Block 3: Conv(128→256) ×2 → BatchNorm → ReLU → MaxPool → Dropout(0.25)  [8→4]
  ├── Block 4: Conv(256→512) → BatchNorm → ReLU → AdaptiveAvgPool              [4→1]
  │
  ├── Flatten → Dropout(0.5) → FC(512→256) → ReLU
  └── Dropout(0.5) → FC(256→10) → Output
```

**Design Decisions:**
- **Kaiming He initialisation** for stable gradient flow through deep ReLU networks
- **Dual convolutions per block** to increase receptive field before downsampling
- **Global Average Pooling** (Block 4) eliminates large FC layers, reducing overfitting
- **Aggressive dropout** (0.25 in conv blocks, 0.5 in classifier) for regularisation

### MobileNetV2 — Transfer Learning (Frozen ImageNet Backbone)

```text
Pretrained MobileNetV2 (ImageNet — 1.2M images, 1000 classes — FROZEN)
  │
  └── Classifier Head: Dropout(0.2) → Linear(1280 → 10)   ◀ Only trainable layer
```

**Strategy:** Freeze the entire feature extraction backbone (2.2M params) and train only a lightweight classifier head (12,810 params). Depthwise separable convolutions reduce computational cost by ~8-9x compared to standard convolutions, making them practical for edge deployment.

### ResNet-18 — Transfer Learning (Frozen ImageNet Backbone)

```text
Pretrained ResNet-18 (ImageNet — 1.2M images, 1000 classes — FROZEN)
  │
  └── FC Head: Linear(512 → 10)   ◀ Only trainable layer (5,130 params)
```

**Strategy:** Standard ResNet-18 (11.2M total params) with every backbone parameter frozen. Only the final fully-connected layer is replaced and trained. Achieves 82.10% accuracy with the fewest trainable parameters of any deployed model, demonstrating that even a simple linear probe on top of strong pretrained features is highly effective.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🔍 Error Analysis & Confusion Patterns

Both models consistently confuse visually similar classes — but MobileNetV2 makes significantly fewer mistakes:

<div align="center">

| Confusion Pair | Custom CNN Errors | MobileNetV2 Errors | Error Reduction | Root Cause |
|:--------------|:-----------------:|:-------------------:|:---------------:|:-----------|
| 🚚 Truck ↔ 🚗 Automobile | 432 | 97 | **78%** | Rectangular vehicle shapes at 32×32 |
| 🚢 Ship ↔ ✈️ Airplane | 375 | 83 | **78%** | Shared sky/water backgrounds |
| 🐱 Cat ↔ 🐕 Dog | 333 | 243 | **27%** | Similar fur texture — even ImageNet struggles |
| 🐴 Horse ↔ 🐕 Dog | 293 | 68 | **77%** | Quadruped body shape overlap |
| 🐦 Bird ↔ 🦌 Deer | 180 | 78 | **57%** | Challenging at low resolution |

</div>

> Measured empirically from the confusion matrices of both models on the full 10 000-image test set (2026-04-10). Each count is the symmetric off-diagonal sum (e.g. cat→dog + dog→cat). Cat↔dog remains the stubbornest pair: the Custom CNN confuses them 333 times, and even the pretrained backbone still trips up 243 times — fine-grained mammal discrimination at 32×32 upscaled to 224×224 is genuinely hard.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Key Features

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║  🧠  5 architectures benchmarked in notebook (3 deployed in live demo)       ║
║  📈  Full training pipeline with cosine annealing LR & progressive unfreezing║
║  🎲  Advanced augmentation: RandomCrop, CutOut, MixUp, CutMix                ║
║  🔬  Grad-CAM interpretability — see what the model actually looks at        ║
║  ⚡  INT8 dynamic quantisation for deployment-ready performance               ║
║  📊  Confusion matrices, training curves, and efficiency benchmarks          ║
║  🖥️  Gradio demo on HF Spaces — interactive side-by-side model comparison    ║
║  🛠️  CLI inference tools — single image, batch, or CIFAR-10 test samples     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🏗️ Technical Architecture & Training Configuration

### Runtime Stack

| Layer | Technology |
|:------|:-----------|
| **Deep Learning** | PyTorch 2.0+ |
| **Pretrained Models** | torchvision (MobileNetV2, ResNet-18, EfficientNet-B0) |
| **Dataset** | CIFAR-10 (60K images, 10 classes) |
| **Evaluation** | scikit-learn (classification reports, confusion matrices) |
| **Visualization** | Matplotlib, Seaborn |
| **Interpretability** | Grad-CAM (custom implementation with PyTorch hooks) |
| **Demo App** | Gradio (Hugging Face Spaces) |
| **Environment** | Jupyter Notebook, Python 3.11 |
| **Hardware** | Auto-detected: CUDA / Apple Silicon MPS / CPU |

### Training Hyperparameters

Both models were trained with **identical hyperparameters** — the only variable is the architecture:

```yaml
Optimiser      : Adam
Learning Rate   : 0.001 (with Cosine Annealing decay)
Weight Decay    : 1e-4
Batch Size      : 128
Epochs          : 15
Loss Function   : CrossEntropyLoss
Training Set    : 50,000 images (full CIFAR-10 training split)
Test Set        : 10,000 images (full CIFAR-10 test split)
Augmentation    : RandomCrop(32,4), HFlip, CutOut(16), MixUp, CutMix
Random Seed     : 42 (full reproducibility)
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📓 Notebook Walkthrough — 14-Section ML Pipeline

The Jupyter notebook follows a structured, production-style ML pipeline:

| # | Section | Description |
|:-:|:--------|:------------|
| 1 | **Environment & Config** | Seed setup, device detection (CUDA/MPS/CPU), hyperparameter config |
| 2 | **Data Preparation & Augmentation** | Full dataset loading with RandomCrop, CutOut, and HFlip augmentation |
| 3 | **MixUp & CutMix** | Batch-level augmentation with Beta-distributed blending |
| 4 | **Model Architectures** | Custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0, Vision Transformer |
| 5 | **Training Pipeline** | Unified training loop with cosine annealing LR, MixUp/CutMix, AMP support |
| 6 | **Train All Models** | Train all 5 architectures with identical hyperparameters |
| 7 | **Progressive Unfreezing** | 3-phase MobileNetV2 fine-tuning (head → partial → full backbone) |
| 8 | **Test Set Evaluation** | Full test set (10K images) accuracy and per-class metrics |
| 9 | **Confusion Matrices** | Side-by-side heatmaps revealing class-level error patterns |
| 10 | **Training Curves** | Loss, accuracy, and LR schedule visualization |
| 11 | **Error Analysis** | Misclassification deep-dive and confusion pair identification |
| 12 | **Efficiency Benchmarks** | Parameters, model size, inference speed, and FPS throughput |
| 13 | **Model Quantization** | INT8 dynamic quantization with size reduction and speedup analysis |
| 14 | **Save Artifacts** | Export config, results metadata, and model checkpoints |

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- GPU recommended (CUDA or Apple Silicon MPS) but not required

### Installation

```bash
# Clone the repository
git clone https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification.git
cd CIFAR-10-Image-Classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook "cifar10 image classification.ipynb"
```

> The CIFAR-10 dataset is downloaded automatically on the first run via `torchvision.datasets`. GPU (CUDA) and Apple Silicon (MPS) acceleration are auto-detected.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 💻 Usage

### Inference (CLI)

```bash
# Classify random CIFAR-10 test images
python predict.py --test-samples 10 --model both

# Classify a single image
python predict.py --image path/to/image.png --model mobilenet

# Classify a directory of images
python predict.py --image-dir path/to/images/ --model both --save results/predictions.png
```

### Grad-CAM Visualisations (CLI)

```bash
# Generate Grad-CAM heatmaps for both models
python gradcam.py --model both --num-images 6

# Visualise specific test images and save output
python gradcam.py --model both --image-index 0 42 100 --save results/gradcam/
```

### 🖥️ Gradio Demo App

```bash
# Run the Gradio demo locally
python app.py
```

Upload any image or click an example from the CIFAR-10 test set for **interactive side-by-side model comparison** with confidence rankings and top-k predictions.

The live demo is hosted on **[Hugging Face Spaces](https://huggingface.co/spaces/mrpouyaalavi/CIFAR-10-Image-Classification)**.

> **Legacy Streamlit landing page:** The original Streamlit URL ([cifar10-pouyaalavi.streamlit.app](https://cifar10-pouyaalavi.streamlit.app/)) is preserved as a lightweight redirect page. To run it locally: `streamlit run streamlit_app.py`

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📁 Project Structure

```text
CIFAR-10-Image-Classification/
│
├── app.py                                # Gradio demo (HF Spaces entry point)
├── streamlit_app.py                      # Streamlit landing page (legacy URL)
├── model_utils.py                        # Shared model architectures & inference
├── benchmark_data.py                     # Canonical metrics (single source of truth)
├── predict.py                            # CLI inference — single / batch / directory
├── gradcam.py                            # Grad-CAM interpretability visualisations
│
├── cifar10 image classification.ipynb    # Main notebook — full 14-section ML pipeline
│
├── checkpoints/                          # Model checkpoints
│   ├── custom_cnn_best.pth
│   ├── mobilenetv2_best.pth
│   └── ...
├── results/                              # Training results & analysis
│   └── training_metadata.json
├── artifacts/
│   └── run_config.json                   # Training hyperparameters
├── examples/                             # Example images for Gradio demo (auto-generated)
├── data/                                 # CIFAR-10 dataset (auto-downloaded)
│
├── requirements.txt                      # Gradio / HF Spaces dependencies
├── requirements-streamlit.txt            # Streamlit landing page dependencies
├── requirements-dev.txt                  # Dev dependencies (pytest)
├── LICENSE                               # MIT License
└── .gitignore                            # Git ignore rules
```

> Model weights are hosted on the [Hugging Face Hub](https://huggingface.co/mrpouyaalavi/cifar10-models) and downloaded automatically at runtime. Datasets (`data/`) are fetched via torchvision on first run.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 💡 Key Takeaways

1. **Transfer learning is remarkably efficient** — A frozen MobileNetV2 backbone with a linear classifier (12K trainable params) outperforms a fully-trained 2.4M-parameter CNN by **+38.5 percentage points**.

2. **Pretrained features generalize across domains** — Despite the resolution gap (ImageNet 224×224 vs. CIFAR-10 32×32), learned representations transfer effectively with proper resizing.

3. **More parameters ≠ better performance** — The Custom CNN has 192× more trainable parameters yet achieves significantly lower accuracy under the same training budget.

4. **Data efficiency matters** — With limited training data, transfer learning reaches production-grade accuracy in a single epoch while training from scratch barely starts to converge.

5. **Speed vs. accuracy trade-off** — The Custom CNN is ~12× faster on CPU (1.38 ms vs 17.22 ms) due to native 32×32 input — relevant for latency-critical edge deployments where you can tolerate lower accuracy for 12× more headroom.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📜 License

Released under the **MIT License** — an OSI-approved, permissive open-source license. See [`LICENSE`](LICENSE) for details.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

<div align="center">

### `> ping --author`

```text
> Target     : Pouya Alavi Naeini — AI & Full-Stack Developer
> University : Macquarie University, Sydney, NSW
> Major      : B.IT — Artificial Intelligence & Web/App Development
> Status     : [●] ONLINE — open to grad & junior opportunities
```

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor=0f172a)](https://huggingface.co/spaces/mrpouyaalavi/CIFAR-10-Image-Classification)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-EE4C2C?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0f172a)](https://www.linkedin.com/in/pouya-alavi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-F7931E?style=for-the-badge&logo=github&logoColor=ffffff&labelColor=0f172a)](https://github.com/mrpouyaalavi)
[![Email](https://img.shields.io/badge/Email-Contact-f59e0b?style=for-the-badge&logo=gmail&logoColor=09090b&labelColor=0f172a)](mailto:pouya@pouyaalavi.dev)

<br/>

**Built with PyTorch & Gradio** · Deployed on Hugging Face Spaces · Designed for Learning, Research & Demonstration

</div>
