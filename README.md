<div align="center">

<<<<<<< HEAD
[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2800&pause=700&color=EE4C2C&center=true&vCenter=true&width=920&lines=CIFAR-10+Deep+Learning+Image+Classification;Custom+CNN+vs+MobileNetV2+vs+ResNet-18;PyTorch+%C2%B7+Grad-CAM+%C2%B7+Gradio+%C2%B7+Hugging+Face+Spaces;5+Architectures+in+Notebook+%C2%B7+3+Deployed+Models+%C2%B7+87.48%25+Accuracy)](https://readme-typing-svg.demolab.com)

![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Demo_App-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD21E?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
=======
# CIFAR-10 Image Classification

**A Comparative Deep Learning Study: Custom CNN vs. Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

</div>

---

An end-to-end deep learning pipeline that designs, trains, and rigorously evaluates **five architectures** on the CIFAR-10 benchmark — including a custom CNN from scratch, transfer learning with MobileNetV2/ResNet-18/EfficientNet-B0, and a Vision Transformer. The project features data augmentation (RandomCrop, CutOut, MixUp, CutMix), cosine annealing LR scheduling, progressive unfreezing, INT8 model quantization, Grad-CAM interpretability, CLI inference tools, and a Streamlit demo app.

<<<<<<< HEAD
# CIFAR-10 Image Classification — From Training to Deployment

> **An end-to-end deep learning project that designs, trains, and evaluates multiple architectures on the CIFAR-10 benchmark, demonstrating the effectiveness of transfer learning compared with a custom CNN baseline across multiple architectures.**

This project is a portfolio-grade machine learning study that goes beyond model training. It includes data augmentation pipelines (RandomCrop, CutOut, MixUp, CutMix), cosine annealing learning rate scheduling, progressive unfreezing, INT8 model quantisation, Grad-CAM interpretability visualisations, CLI inference tools, and a Gradio demo deployed on Hugging Face Spaces — all documented in a structured Jupyter notebook.

**[📓 Explore the Notebook](cifar10%20image%20classification.ipynb)** &nbsp;·&nbsp; **[🚀 Live Demo](https://mrpouyaalavi-cifar-10-image-classification.hf.space)** &nbsp;·&nbsp; **[📊 Key Results](#-key-results--performance-benchmarks)**
=======
> **[Explore the Notebook](cifar10%20image%20classification.ipynb)**

---

## Table of Contents

- [Motivation](#motivation)
- [Key Results](#key-results)
- [Model Architectures](#model-architectures)
- [Error Analysis](#error-analysis)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Training Configuration](#training-configuration)
- [Tech Stack](#tech-stack)
- [Key Takeaways](#key-takeaways)
- [Roadmap](#roadmap)
- [License](#license)

---
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

## Motivation

> _How much does a pretrained backbone actually help compared to training from scratch when models are evaluated under controlled conditions?_

<<<<<<< HEAD
Deep learning practitioners often default to transfer learning without quantifying its advantage under comparable settings. This project answers that question through a controlled experiment: identical dataset, optimiser family, learning-rate scheduling, epoch budget, and augmentation strategy, with architecture and transfer-learning strategy as the key variables.
=======
This project answers that question through a **controlled experiment**: identical dataset, optimizer, learning rate, and epoch count — the only variable is the architecture and whether the weights are pretrained. The results have direct implications for real-world model selection and deployment strategy.
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

---

<<<<<<< HEAD
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
=======
## Key Results

<div align="center">

| Metric | Custom CNN | MobileNetV2 | Winner |
|:-------|:----------:|:-----------:|:------:|
| **Test Accuracy** | 55.96% | **85.53%** | MobileNetV2 |
| **Trainable Params** | 2,462,282 | **12,810** | MobileNetV2 |
| **Model Size** | 9.40 MB | **8.66 MB** | MobileNetV2 |
| **Inference Latency** | **0.84 ms** | 5.33 ms | Custom CNN |
| **Throughput** | **~1,191 FPS** | ~188 FPS | Custom CNN |

</div>

> **Key Finding:** MobileNetV2 achieves 85.53% accuracy with just 0.5% of the Custom CNN's trainable parameters. Transfer learning doesn't just win on accuracy — it wins with **200x fewer trainable weights**.

### Training Progression
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

| Epoch | Custom CNN (Val Acc) | MobileNetV2 (Val Acc) |
|:-----:|:--------------------:|:---------------------:|
| 1 | 27.1% | **83.0%** |
| 2 | 36.8% | 84.6% |
| 3 | 44.8% | 85.4% |
| 4 | 49.1% | 85.5% |
| 5 | 56.0% | **85.5%** |

<<<<<<< HEAD
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
=======
MobileNetV2 reaches 83% after **a single epoch**. The custom CNN is still at 27% — demonstrating the enormous advantage of pretrained feature representations.

---

## Model Architectures

### Custom CNN — 4-Block Design (From Scratch)

A purpose-built convolutional network with progressive channel expansion and regularization:

```
Input (3 x 32 x 32)
  |
  +-- Block 1: Conv(3->64) x2 -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)     [32->16]
  +-- Block 2: Conv(64->128) x2 -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)   [16->8]
  +-- Block 3: Conv(128->256) x2 -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)  [8->4]
  +-- Block 4: Conv(256->512) -> BatchNorm -> ReLU -> AdaptiveAvgPool               [4->1]
  |
  +-- Flatten -> Dropout(0.5) -> FC(512->256) -> ReLU
  +-- Dropout(0.5) -> FC(256->10) -> Output
```

**Design decisions:**
- **Kaiming He initialization** for stable gradient flow through deep ReLU networks
- **Dual convolutions per block** for richer feature extraction before downsampling
- **Global average pooling** (Block 4) to reduce spatial dimensions without FC overhead
- **Aggressive dropout** (0.25/0.5) to mitigate overfitting on the 10K training subset
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

### MobileNetV2 — Transfer Learning (Frozen Backbone)

<<<<<<< HEAD
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
=======
```
Pretrained MobileNetV2 (ImageNet weights — FROZEN)
  |
  +-- Classifier Head: Dropout(0.2) -> Linear(1280 -> 10)   <-- only trainable layer
```

**Strategy:** Freeze the entire feature extraction backbone (2.2M params) and train only a lightweight classifier head (12,810 params). Input images are resized from 32x32 to 224x224 to match the expected ImageNet resolution.

---

## Error Analysis
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

Both models consistently confuse visually similar classes, but MobileNetV2 makes significantly fewer mistakes:

| Confusion Pair | Custom CNN Errors | MobileNetV2 Errors | Reduction | Root Cause |
|:--------------|:-----------------:|:-------------------:|:---------:|:-----------|
| Cat <-> Dog | 431 | 177 | **59%** | Similar fur texture and body structure |
| Horse <-> Dog | 231 | 75 | **68%** | Quadruped body shape overlap |
| Ship <-> Airplane | 226 | 55 | **76%** | Shared sky/water backgrounds |
| Truck <-> Automobile | 220 | 53 | **76%** | Rectangular vehicle shapes at 32x32 |
| Bird <-> Deer | 219 | 49 | **78%** | Challenging at low resolution |

<<<<<<< HEAD
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
=======
---

## Getting Started
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

### Prerequisites

- Python 3.11 (as specified in `runtime.txt`)
- pip or conda
<<<<<<< HEAD
- GPU recommended, but not required
=======
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

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

<<<<<<< HEAD
> The CIFAR-10 dataset is downloaded automatically on first run via `torchvision.datasets`.
=======
> The CIFAR-10 dataset is downloaded automatically on first run via `torchvision.datasets`. GPU (CUDA) and Apple Silicon (MPS) acceleration are auto-detected.
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

---

## Usage

### CLI Inference

```bash
python predict.py --test-samples 10 --model all
python predict.py --image path/to/image.png --model mobilenet
python predict.py --image-dir path/to/images/ --model all --save results/predictions.png
```

<<<<<<< HEAD
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
=======
### Grad-CAM Visualizations (CLI)

```bash
# Generate Grad-CAM heatmaps for both models
python gradcam.py --model both --num-images 6

# Visualize specific test images and save output
python gradcam.py --model both --image-index 0 42 100 --save results/gradcam/
```

### Streamlit Demo App

```bash
streamlit run app.py
```

Upload any image or sample from the CIFAR-10 test set for interactive side-by-side model comparison with confidence bars.

---

## Project Structure
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

```
CIFAR-10-Image-Classification/
<<<<<<< HEAD
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
=======
|
|-- cifar10 image classification.ipynb   # Main notebook — full ML pipeline (14 sections)
|-- predict.py                            # CLI inference — single / batch / directory
|-- gradcam.py                            # Grad-CAM interpretability visualizations
|-- app.py                                # Streamlit demo app — interactive classification
|
|-- artifacts/
|   +-- run_config.json                   # Training hyperparameters & config
|-- results/
|   +-- training_metadata.json            # Experiment results & metrics
|
|-- requirements.txt                      # Python dependencies
|-- LICENSE                               # MIT License
+-- .gitignore                            # Git ignore rules
```

> Model weights (`.pth`), datasets (`data/`), and checkpoints (`checkpoints/`) are excluded via `.gitignore` and regenerated automatically when running the notebook.
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

---

## Notebook Walkthrough

The notebook follows a structured ML pipeline across **14 documented sections**:

| # | Section | Description |
|:-:|---------|:------------|
| 1 | **Environment & Config** | Seed setup, device detection, hyperparameter configuration |
| 2 | **Data Preparation & Augmentation** | Full dataset loading with RandomCrop, CutOut, and HFlip augmentation |
| 3 | **MixUp & CutMix** | Batch-level augmentation with Beta-distributed blending |
| 4 | **Model Architectures** | Custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0, and Vision Transformer |
| 5 | **Training Pipeline** | Unified loop with cosine annealing LR, MixUp/CutMix, and AMP support |
| 6 | **Train All Models** | Train all 5 architectures with identical hyperparameters |
| 7 | **Progressive Unfreezing** | 3-phase MobileNetV2 fine-tuning (head → partial → full backbone) |
| 8 | **Test Set Evaluation** | Full test set (10K images) accuracy and per-class metrics |
| 9 | **Confusion Matrices** | Side-by-side heatmaps revealing class-level error patterns |
| 10 | **Training Curves** | Loss, accuracy, and LR schedule visualisation |
| 11 | **Error Analysis** | Misclassification deep-dive and confusion pair identification |
| 12 | **Efficiency Benchmarks** | Parameters, model size, inference speed, and FPS throughput |
| 13 | **Model Quantization** | INT8 dynamic quantization with size reduction and speedup analysis |
| 14 | **Save Artifacts** | Export config, results metadata, and model checkpoints |

<<<<<<< HEAD
1. **Transfer learning is highly efficient** — MobileNetV2 strongly outperforms a custom CNN while training only a tiny fraction of the parameters.
2. **Pretrained features transfer well** — even from ImageNet-scale pretraining to CIFAR-10.
3. **More trainable parameters do not guarantee better results** under a fixed training budget.
4. **Data efficiency matters** — transfer learning reaches strong performance within a very small number of epochs.
5. **Speed vs. accuracy trade-offs remain real** — the custom CNN is ~10× faster on CPU, while ResNet-18 (87.48%) and MobileNetV2 (86.91%) trade a little latency for ~+39 pp of accuracy.
=======
---

## Training Configuration

Both models were trained with **identical hyperparameters** — the only variable is the architecture and pretrained weights:

```yaml
Optimizer       : Adam
Learning Rate   : 0.001 (with Cosine Annealing decay)
Weight Decay    : 1e-4
Batch Size      : 128
Epochs          : 15
Loss Function   : CrossEntropyLoss
Training Set    : 50,000 images (full CIFAR-10 training split)
Test Set        : 10,000 images (full CIFAR-10 test split)
Augmentation    : RandomCrop(32,4), HFlip, CutOut(16), MixUp, CutMix
Random Seed     : 42 (full reproducibility)
Device          : Apple M-series GPU (MPS) / CUDA / CPU
```

---

## Tech Stack

| Category | Technologies |
|:---------|:------------|
| **Deep Learning** | PyTorch 2.0+ |
| **Pretrained Models** | torchvision (MobileNetV2, ResNet-18, EfficientNet-B0) |
| **Dataset** | CIFAR-10 (60K images, 10 classes) |
| **Evaluation** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Interpretability** | Grad-CAM (custom implementation) |
| **Demo App** | Streamlit |
| **Environment** | Jupyter Notebook, Python 3.11 |
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

---

## Key Takeaways

1. **Transfer learning is remarkably efficient** — A frozen MobileNetV2 backbone with a linear classifier (12K trainable params) outperforms a fully-trained 2.4M-parameter CNN by ~30 percentage points.

2. **Pretrained features generalize across domains** — Despite the resolution gap (ImageNet 224x224 vs. CIFAR-10 32x32), learned representations transfer effectively with proper resizing.

<<<<<<< HEAD
Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
=======
3. **More parameters ≠ better performance** — The Custom CNN has 200x more trainable parameters yet achieves significantly lower accuracy under the same training budget.
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

4. **Data efficiency matters** — With just 1K images per class, transfer learning reaches production-grade accuracy while training from scratch barely starts to converge.

5. **Speed vs. accuracy trade-off** — The Custom CNN is 6x faster at inference (0.84ms vs 5.33ms) due to native 32x32 input — relevant for latency-critical edge deployments.

---

## Roadmap

- [x] ~~Train with the **full CIFAR-10 dataset** (50K images) and more epochs~~ (15 epochs, full 50K)
- [x] ~~Add **learning rate scheduling** (cosine annealing)~~ (CosineAnnealingLR)
- [x] ~~Experiment with **progressive unfreezing** of MobileNetV2 backbone layers~~ (3-phase)
- [x] ~~Implement **data augmentation** (RandomCrop, CutOut, MixUp, CutMix)~~
- [x] ~~Benchmark additional architectures (ResNet-18, EfficientNet-B0, ViT)~~
- [x] ~~Add **model quantization** (INT8) for deployment-ready performance~~
- [x] ~~Build a **Streamlit demo app** for interactive classification~~ (`app.py`)
- [x] ~~Grad-CAM visualizations~~ (`gradcam.py`)
- [x] ~~CLI inference script~~ (`predict.py`)

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

<<<<<<< HEAD
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
=======
**Built with PyTorch** · Designed for Learning & Demonstration
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

</div>
