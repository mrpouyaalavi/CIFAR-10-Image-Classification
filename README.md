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

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2800&pause=700&color=EE4C2C&center=true&vCenter=true&width=920&lines=CIFAR-10+Deep+Learning+Image+Classification;Custom+CNN+vs+MobileNetV2+vs+ResNet-18;PyTorch+%C2%B7+Grad-CAM+%C2%B7+Gradio+%C2%B7+Hugging+Face+Spaces;5+Architectures+in+Notebook+%C2%B7+3+Deployed+Models+%C2%B7+87.48%25+Accuracy)](https://readme-typing-svg.demolab.com)

![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-5.29.0-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD21E?style=for-the-badge)
![Tests](https://img.shields.io/badge/79_Tests-Pytest-6E9F18?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

# CIFAR-10 Image Classification — From Training to Deployment

> **An end-to-end deep learning portfolio project that trains, evaluates, and compares multiple architectures on the CIFAR-10 dataset, demonstrating the practical impact of transfer learning over a custom CNN baseline — and packaging the results into a live interactive demo.**

This project goes beyond model training. It includes augmentation pipelines, cosine annealing scheduling, progressive unfreezing, INT8 quantisation experiments (notebook), Grad-CAM interpretability visualisations, CLI inference tools, and a Gradio demo deployed on Hugging Face Spaces — all documented in a structured Jupyter notebook.

**[📓 Explore the Notebook](cifar10%20image%20classification.ipynb)** &nbsp;·&nbsp; **[🚀 Live Demo](https://cifar10.pouyaalavi.dev)** &nbsp;·&nbsp; **[📊 Key Results](#-key-results--performance-benchmarks)**

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🎯 Research Question

> _How much does a pretrained backbone actually help compared to training from scratch when both models share the same training setup?_

A common assumption in deep learning is that transfer learning always wins — but by how much, and under what conditions? This project answers that question with a controlled comparison: the same dataset, optimiser, learning-rate schedule, epoch budget, and augmentation pipeline across all models, with architecture and pretraining strategy as the only variables.

The results are relevant to:

- **Model selection** in resource-constrained and edge-deployment environments
- **Training efficiency** when labelled data or compute budget is limited
- **Deployment strategy** when balancing latency, model size, and accuracy

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Screenshots

<div align="center">

| Live Demo | Model Comparison |
|:---:|:---:|
| <img width="500" alt="Live Demo — upload interface with model selector" src="assets/screenshot_live_demo.png"/> | <img width="500" alt="Model Comparison — benchmark table and key findings" src="assets/screenshot_model_comparison.png"/> |

</div>


<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## ✨ Key Features

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║  🧠  3 deployed models in live demo · 5 architectures explored in notebook   ║
║  📈  Full training pipeline with cosine annealing and progressive unfreezing ║
║  🎲  Advanced augmentation: RandomCrop, CutOut, MixUp, CutMix                ║
║  🔬  Grad-CAM interpretability for visual model explanations                 ║
║  ⚡  INT8 dynamic quantisation experiments for deployment analysis            ║
║  📊  Confusion matrices, training curves, and efficiency benchmarks          ║
║  🖥️  Gradio demo on HF Spaces · weights served from HF Hub at runtime        ║
║  🛠️  CLI inference for single image, batch, and full test-set evaluation     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📊 Key Results & Performance Benchmarks

<div align="center">

| Metric | Custom CNN | MobileNetV2 | ResNet-18 | Winner |
|:-------|:----------:|:-----------:|:---------:|:------:|
| **Test Accuracy** | 48.40% | 86.91% | **87.48%** | 🏆 ResNet-18 |
| **Trainable Params** | 2,462,282 | 12,810 | **5,130** | 🏆 ResNet-18 |
| **Model Size** | 9.42 MB | **8.76 MB** | 44.80 MB | 🏆 MobileNetV2 |
| **CPU Latency (batch 1)** | **1.38 ms** | 17.22 ms | 9.80 ms | 🏆 Custom CNN |
| **Throughput** | **~724 FPS** | ~58 FPS | ~102 FPS | 🏆 Custom CNN |

</div>

> Accuracy results were evaluated on the 10,000-image CIFAR-10 test set. Latency and throughput figures are hardware-dependent and should be interpreted as project benchmark results rather than universal model performance. ResNet-18 retrained and verified 2026-04-18 via cached-features linear probe.

> **Key finding:** In this experiment, ResNet-18 achieved **87.48% accuracy** with just **0.2%** of the Custom CNN's trainable parameters — a **+39.1 percentage-point** lift for a **480× reduction in trainable weights**. MobileNetV2 landed within a fraction of a point at **86.91%** with a different parameter/latency trade-off.

### Training Progression — Convergence Comparison

```text
Epoch   Custom CNN (Val Acc)     MobileNetV2 (Val Acc)     ResNet-18 (Val Acc)
─────   ────────────────────     ─────────────────────     ───────────────────
  1          21.0%                    85.88%                    84.21%
  2          27.8%                    86.80%                    85.64%
  3          32.4%                    86.91%                    86.27%
  …            …                        …                          …
 15          48.40%                   86.91%                    87.16%
 30           —                        —                        87.48%
```

Both transfer-learning models reach strong accuracy within 1–3 epochs because their frozen backbones already encode powerful ImageNet features. The Custom CNN is still improving across the full 15-epoch budget — highlighting the data efficiency of pretrained representations.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🏗️ Technical Architecture & Training Configuration

### Runtime Stack

| Layer | Technology |
|:------|:-----------|
| **Framework** | PyTorch 2.0+ |
| **Deployed Models** | torchvision (MobileNetV2, ResNet-18) + Custom CNN |
| **Notebook-only Models** | EfficientNet-B0, ViT (Small) — explored in notebook, not deployed |
| **Dataset** | CIFAR-10 — 60K images, 10 classes |
| **Evaluation** | scikit-learn (classification reports, confusion matrices) |
| **Visualization** | Matplotlib (deployed) · Seaborn (notebook) |
| **Interpretability** | Grad-CAM with PyTorch hooks |
| **Demo App** | Gradio 5.29.0 on Hugging Face Spaces |
| **Model Weights** | Hugging Face Hub (`mrpouyaalavi/cifar10-models`) |
| **Hardware** | Auto-detected: CUDA / Apple Silicon MPS / CPU |

### Training Hyperparameters

```yaml
Optimiser      : Adam
Learning Rate  : 0.001  (with Cosine Annealing decay)
Weight Decay   : 1e-4
Batch Size     : 128
Epochs         : 15  (ResNet-18 linear probe: 30)
Loss Function  : CrossEntropyLoss
Training Set   : 50,000 images
Test Set       : 10,000 images
Augmentation   : RandomCrop(32,4), HFlip, CutOut(16), MixUp, CutMix
Random Seed    : 42
```

### Model Architectures

**Custom CNN — 4-Block Design (trained from scratch)**
```text
Input (3 × 32 × 32)
  ├── Block 1: Conv(3→64)×2   → BN → ReLU → MaxPool → Dropout(0.25)
  ├── Block 2: Conv(64→128)×2 → BN → ReLU → MaxPool → Dropout(0.25)
  ├── Block 3: Conv(128→256)×2 → BN → ReLU → MaxPool → Dropout(0.25)
  ├── Block 4: Conv(256→512)  → BN → ReLU → AdaptiveAvgPool
  └── Flatten → Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.5) → FC(256→10)
```

**MobileNetV2 — frozen ImageNet backbone + `Dropout(0.2) → Linear(1280→10)`**

**ResNet-18 — frozen ImageNet backbone + `Linear(512→10)` (5,130 trainable params)**

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🔍 Error Analysis & Confusion Patterns

<div align="center">

| Confusion Pair | Custom CNN | MobileNetV2 | Reduction | Root Cause |
|:--------------|:-----------:|:-----------:|:---------:|:-----------|
| 🚚 Truck ↔ 🚗 Automobile | 432 | 97 | **78%** | Similar vehicle structure at 32×32 |
| 🚢 Ship ↔ ✈️ Airplane | 375 | 83 | **78%** | Shared background cues |
| 🐱 Cat ↔ 🐕 Dog | 333 | 243 | **27%** | Fine-grained mammal similarity |
| 🐴 Horse ↔ 🐕 Dog | 293 | 68 | **77%** | Quadruped shape overlap |
| 🐦 Bird ↔ 🦌 Deer | 180 | 78 | **57%** | Challenging low-resolution silhouettes |

</div>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📓 Notebook Walkthrough — 14-Section ML Pipeline

| # | Section | Description |
|:-:|:--------|:------------|
| 1 | **Environment & Config** | Seed setup, device detection, hyperparameter config |
| 2 | **Data Preparation & Augmentation** | Dataset loading and augmentation pipeline |
| 3 | **MixUp & CutMix** | Batch-level augmentation experiments |
| 4 | **Model Architectures** | Custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0, ViT |
| 5 | **Training Pipeline** | Unified loop with cosine annealing and AMP support |
| 6 | **Train All Models** | Controlled comparisons across architectures |
| 7 | **Progressive Unfreezing** | MobileNetV2 fine-tuning schedule |
| 8 | **Test Set Evaluation** | Full test-set accuracy and class-level metrics |
| 9 | **Confusion Matrices** | Side-by-side error analysis |
| 10 | **Training Curves** | Loss, accuracy, and LR schedule visualisation |
| 11 | **Error Analysis** | Misclassification deep-dive |
| 12 | **Efficiency Benchmarks** | Parameters, size, latency, and throughput |
| 13 | **Model Quantization** | INT8 quantisation experiments |
| 14 | **Save Artifacts** | Export config, results, and metadata |

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🎯 Project Governance

### License
Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

### Maintainers

| Name | Role |
|:-----|:-----|
| Pouya Alavi Naeini | Lead — ML pipeline, deployment, Gradio app |

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Repository Layout

```text
app.py                             Gradio demo — HF Spaces entry point
model_utils.py                     Shared model architectures & inference
benchmark_data.py                  Canonical benchmark metrics (single source of truth)
predict.py                         CLI inference tools
gradcam.py                         Grad-CAM visualisations

cifar10 image classification.ipynb Main 14-section notebook

scripts/                           Retraining & measurement scripts
  retrain_custom_cnn.py
  retrain_mobilenetv2.py
  retrain_resnet18.py
  retrain_resnet18_fast.py         Cached-features linear probe (fast retraining)
  measure_model.py

tests/                             Pytest unit & integration tests (79 tests)
  conftest.py  test_models.py  test_inference.py  test_preprocessing.py
  test_gradcam.py  test_benchmark_data.py  test_checkpoint_remap.py  test_device.py

results/                           Training results, confusion matrices, metadata
examples/                          Example CIFAR-10 images for the live demo
data/                              CIFAR-10 dataset (auto-downloaded via torchvision)

requirements.txt                   HF Spaces / Gradio dependencies
requirements-dev.txt               Development dependencies
runtime.txt                        Python version pin
pytest.ini                         Pytest configuration
```

> Model weights are hosted on [Hugging Face Hub](https://huggingface.co/mrpouyaalavi/cifar10-models) and downloaded automatically at runtime — no binaries in this repo.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Quick Start

### Prerequisites
- Python `3.11` (as specified in `runtime.txt`)
- pip or conda
- GPU recommended but not required — CPU works fine for inference

### Setup

```bash
# Clone and install
git clone https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification.git
cd CIFAR-10-Image-Classification

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook "cifar10 image classification.ipynb"
# CIFAR-10 dataset downloads automatically on first run via torchvision
```

### CLI Inference

```bash
python predict.py --test-samples 10 --model all
python predict.py --image path/to/image.png --model mobilenet
python predict.py --image-dir path/to/images/ --model all --save results/predictions.png
```

### Grad-CAM Visualisations

```bash
python gradcam.py --model all --num-images 6
python gradcam.py --model all --image-index 0 42 100 --save results/gradcam/
```

### Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q     # 79 tests
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Documentation Map

| Document | Path |
|:---------|:-----|
| Benchmark metrics (single source of truth) | [benchmark_data.py](./benchmark_data.py) |
| Training metadata & retrain history | [results/training_metadata.json](./results/training_metadata.json) |
| ResNet-18 training history | [results/resnet18_training_history.json](./results/resnet18_training_history.json) |
| Main notebook | [cifar10 image classification.ipynb](./cifar10%20image%20classification.ipynb) |
| Live demo | [Hugging Face Spaces](https://mrpouyaalavi-cifar-10-image-classification.hf.space) |
| Model weights | [HF Hub — mrpouyaalavi/cifar10-models](https://huggingface.co/mrpouyaalavi/cifar10-models) |

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Acknowledgements

Built with the support of the open-source community. This project benefits from:

- [PyTorch](https://pytorch.org/) — Deep learning framework and pretrained model weights.
- [Hugging Face](https://huggingface.co/) — Model hosting and Spaces deployment infrastructure.
- [Gradio](https://www.gradio.app/) — Interactive demo framework.

<br/>

<div align="center">

### `> ping --author`

```text
> Target     : Pouya Alavi Naeini — Software Engineer | Applied AI/ML
> University : Macquarie University, Sydney, NSW
> Major      : B.IT — Artificial Intelligence & Web/App Development
> Status     : [●] ONLINE — open to grad & junior opportunities
```

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor=0f172a)](https://mrpouyaalavi-cifar-10-image-classification.hf.space)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-EE4C2C?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0f172a)](https://www.linkedin.com/in/pouya-alavi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-F7931E?style=for-the-badge&logo=github&logoColor=ffffff&labelColor=0f172a)](https://github.com/mrpouyaalavi)
[![Email](https://img.shields.io/badge/Email-Contact-f59e0b?style=for-the-badge&logo=gmail&logoColor=09090b&labelColor=0f172a)](mailto:pouya@pouyaalavi.dev)

<br/>

*CIFAR-10 Image Classification is an independent, open-source portfolio project.*

</div>
