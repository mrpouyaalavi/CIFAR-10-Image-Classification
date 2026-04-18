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
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Demo_App-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD21E?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

# CIFAR-10 Image Classification — From Training to Deployment

> **An end-to-end deep learning project that designs, trains, and evaluates multiple architectures on the CIFAR-10 benchmark, demonstrating the effectiveness of transfer learning compared with a custom CNN baseline across multiple architectures.**

This project is a portfolio-grade machine learning study that goes beyond model training. It includes data augmentation pipelines (RandomCrop, CutOut, MixUp, CutMix), cosine annealing learning rate scheduling, progressive unfreezing, INT8 model quantisation, Grad-CAM interpretability visualisations, CLI inference tools, and a Gradio demo deployed on Hugging Face Spaces — all documented in a structured Jupyter notebook.

**[📓 Explore the Notebook](cifar10%20image%20classification.ipynb)** &nbsp;·&nbsp; **[🚀 Live Demo](https://mrpouyaalavi-cifar-10-image-classification.hf.space)** &nbsp;·&nbsp; **[📊 Key Results](#-key-results--performance-benchmarks)**

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🎯 Motivation & Research Question

> _How much does a pretrained backbone actually help compared to training from scratch when models are evaluated under controlled conditions?_

Deep learning practitioners often default to transfer learning without quantifying its advantage under comparable settings. This project answers that question through a controlled experiment: identical dataset, optimiser family, learning-rate scheduling, epoch budget, and augmentation strategy, with architecture and transfer-learning strategy as the key variables.

The results have direct implications for:

- **Model selection** in resource-constrained environments
- **Training efficiency** when labelled data is limited
- **Deployment strategy** when balancing latency, size, and accuracy

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

> All numbers are measured empirically from the final evaluation checkpoints on the full 10,000-image CIFAR-10 test set (Custom CNN & MobileNetV2 verified 2026-04-10; ResNet-18 retrained and verified 2026-04-18 via cached-features linear probe; Custom CNN latency re-measured 2026-04-11 with 100 trials on Apple silicon CPU, batch size 1).

> **Key finding:** ResNet-18 achieves **87.48% accuracy** with just **0.2%** of the Custom CNN's trainable parameters — a **+39.1 percentage-point** lift for a **480× reduction in trainable weights**. MobileNetV2 lands within a fraction of a point at **86.91%** with a different parameter/latency trade-off.

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

Both transfer-learning models (MobileNetV2 and ResNet-18) reach strong accuracy within 1–3 epochs and plateau quickly, because their frozen backbones already encode powerful ImageNet features. The Custom CNN is still improving across the full 15-epoch budget, highlighting the value of pretrained feature representations.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🧠 Model Architectures

### Custom CNN — 4-Block Design (Trained From Scratch)

```text
Input (3 × 32 × 32)
  │
  ├── Block 1: Conv(3→64) ×2 → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ├── Block 2: Conv(64→128) ×2 → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ├── Block 3: Conv(128→256) ×2 → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ├── Block 4: Conv(256→512) → BatchNorm → ReLU → AdaptiveAvgPool
  │
  ├── Flatten → Dropout(0.5) → FC(512→256) → ReLU
  └── Dropout(0.5) → FC(256→10) → Output
```

**Design decisions:**
- **Kaiming He initialisation** for stable gradient flow
- **Dual convolutions per block** before downsampling
- **Global Average Pooling** to reduce classifier size
- **Aggressive dropout** for regularisation

### MobileNetV2 — Transfer Learning

```text
Pretrained MobileNetV2 backbone (frozen)
  │
  └── Classifier Head: Dropout(0.2) → Linear(1280 → 10)
```

**Strategy:** Freeze the feature extractor and train a lightweight classification head, then apply progressive unfreezing for stronger adaptation.

### ResNet-18 — Transfer Learning

```text
Pretrained ResNet-18 backbone (frozen)
  │
  └── FC Head: Linear(512 → 10)
```

**Strategy:** Replace and train only the final fully connected layer. This model achieves strong accuracy with the fewest trainable parameters among the deployed models.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🔍 Error Analysis & Confusion Patterns

Both models consistently confuse visually similar classes, but MobileNetV2 makes significantly fewer mistakes:

<div align="center">

| Confusion Pair | Custom CNN Errors | MobileNetV2 Errors | Error Reduction | Root Cause |
|:--------------|:-----------------:|:-------------------:|:---------------:|:-----------|
| 🚚 Truck ↔ 🚗 Automobile | 432 | 97 | **78%** | Similar vehicle structure at 32×32 |
| 🚢 Ship ↔ ✈️ Airplane | 375 | 83 | **78%** | Shared background cues |
| 🐱 Cat ↔ 🐕 Dog | 333 | 243 | **27%** | Fine-grained mammal similarity |
| 🐴 Horse ↔ 🐕 Dog | 293 | 68 | **77%** | Quadruped shape overlap |
| 🐦 Bird ↔ 🦌 Deer | 180 | 78 | **57%** | Challenging low-resolution silhouettes |

</div>

> Measured empirically from confusion matrices on the full CIFAR-10 test set.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## ✨ Key Features

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║  🧠  3 deployed models in the live demo · 5 architectures explored in notebook║
║  📈  Full training pipeline with cosine annealing and progressive unfreezing ║
║  🎲  Advanced augmentation: RandomCrop, CutOut, MixUp, CutMix                ║
║  🔬  Grad-CAM interpretability for visual model explanations                 ║
║  ⚡  INT8 dynamic quantisation experiments for deployment analysis            ║
║  📊  Confusion matrices, training curves, and efficiency benchmarks          ║
║  🖥️  Gradio demo on HF Spaces with interactive image classification          ║
║  🛠️  CLI inference tools for single image, batch, and test-set evaluation    ║
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
| **Interpretability** | Grad-CAM with PyTorch hooks |
| **Demo App** | Gradio on Hugging Face Spaces |
| **Environment** | Jupyter Notebook, Python 3.11 |
| **Hardware** | Auto-detected: CUDA / Apple Silicon MPS / CPU |

### Training Hyperparameters

The core training budget was kept consistent across experiments, with architecture-specific adaptations where required:

```yaml
Optimiser      : Adam
Learning Rate  : 0.001 (with Cosine Annealing decay)
Weight Decay   : 1e-4
Batch Size     : 128
Epochs         : 15
Loss Function  : CrossEntropyLoss
Training Set   : 50,000 images
Test Set       : 10,000 images
Augmentation   : RandomCrop(32,4), HFlip, CutOut(16), MixUp, CutMix
Random Seed    : 42
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📓 Notebook Walkthrough — 14-Section ML Pipeline

| # | Section | Description |
|:-:|:--------|:------------|
| 1 | **Environment & Config** | Seed setup, device detection, hyperparameter configuration |
| 2 | **Data Preparation & Augmentation** | Dataset loading and augmentation pipeline |
| 3 | **MixUp & CutMix** | Batch-level augmentation experiments |
| 4 | **Model Architectures** | Custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0, Vision Transformer |
| 5 | **Training Pipeline** | Unified training loop with cosine annealing and AMP support |
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

## 🚀 Getting Started

### Prerequisites

- Python 3.11 (as specified in `runtime.txt`)
- pip or conda
- GPU recommended, but not required

### Installation

```bash
git clone https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification.git
cd CIFAR-10-Image-Classification

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook "cifar10 image classification.ipynb"
```

> The CIFAR-10 dataset is downloaded automatically on first run via `torchvision.datasets`.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 💻 Usage

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

### Gradio Demo App

```bash
python app.py
```

Upload an image or select an example to compare deployed models with live predictions and confidence rankings.

The live demo is hosted on **[Hugging Face Spaces](https://mrpouyaalavi-cifar-10-image-classification.hf.space)**.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📁 Project Structure

```text
CIFAR-10-Image-Classification/
│
├── app.py                                # Gradio demo (HF Spaces entry point)
├── model_utils.py                        # Shared model architectures & inference
├── benchmark_data.py                     # Canonical benchmark metrics
├── predict.py                            # CLI inference tools
├── gradcam.py                            # Grad-CAM visualisations
│
├── cifar10 image classification.ipynb    # Main notebook
│
├── scripts/                              # Retraining & measurement scripts
│   ├── retrain_custom_cnn.py             #   Custom CNN training run
│   ├── retrain_mobilenetv2.py            #   MobileNetV2 frozen-backbone training
│   ├── retrain_resnet18.py               #   ResNet-18 frozen-backbone training
│   └── measure_model.py                  #   Accuracy, latency & confusion pairs
│
├── tests/                                # Pytest unit & integration tests
│   ├── conftest.py                       #   Shared fixtures
│   ├── test_models.py                    #   Architecture smoke tests
│   ├── test_inference.py                 #   predict() contract tests
│   ├── test_preprocessing.py             #   Transform pipeline tests
│   ├── test_gradcam.py                   #   Grad-CAM hook tests
│   ├── test_benchmark_data.py            #   Metric consistency tests
│   ├── test_checkpoint_remap.py          #   Checkpoint key migration tests
│   └── test_device.py                    #   Device selection tests
│
├── results/                              # Training results & analysis
├── artifacts/                            # Exported configs and run metadata
├── examples/                             # Example images for the live demo
├── data/                                 # CIFAR-10 dataset (auto-downloaded)
│
├── requirements.txt                      # Gradio / HF Spaces dependencies
├── requirements-dev.txt                  # Development dependencies
├── runtime.txt                           # Python version pin for HF Spaces
├── pytest.ini                            # Pytest configuration
├── LICENSE                               # MIT License
└── .gitignore                            # Git ignore rules
```

> Model weights are hosted on the [Hugging Face Hub](https://huggingface.co/mrpouyaalavi/cifar10-models) and downloaded automatically at runtime.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 💡 Key Takeaways

1. **Transfer learning is highly efficient** — MobileNetV2 strongly outperforms a custom CNN while training only a tiny fraction of the parameters.
2. **Pretrained features transfer well** — even from ImageNet-scale pretraining to CIFAR-10.
3. **More trainable parameters do not guarantee better results** under a fixed training budget.
4. **Data efficiency matters** — transfer learning reaches strong performance within a very small number of epochs.
5. **Speed vs. accuracy trade-offs remain real** — the custom CNN is ~10× faster on CPU, while ResNet-18 (87.48%) and MobileNetV2 (86.91%) trade a little latency for ~+39 pp of accuracy.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📜 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

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

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor=0f172a)](https://mrpouyaalavi-cifar-10-image-classification.hf.space)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-EE4C2C?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0f172a)](https://www.linkedin.com/in/pouya-alavi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-F7931E?style=for-the-badge&logo=github&logoColor=ffffff&labelColor=0f172a)](https://github.com/mrpouyaalavi)
[![Email](https://img.shields.io/badge/Email-Contact-f59e0b?style=for-the-badge&logo=gmail&logoColor=09090b&labelColor=0f172a)](mailto:pouya@pouyaalavi.dev)

<br/>

**Built with PyTorch & Gradio · Deployed on Hugging Face Spaces · Designed for learning, research, and demonstration**

</div>
