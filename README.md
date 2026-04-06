<div align="center">

# CIFAR-10 Image Classification

**A Comparative Deep Learning Study: Custom CNN vs. Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

An end-to-end deep learning pipeline that designs, trains, and rigorously evaluates **five architectures** on the CIFAR-10 benchmark — including a custom CNN from scratch, transfer learning with MobileNetV2/ResNet-18/EfficientNet-B0, and a Vision Transformer. The project features data augmentation (RandomCrop, CutOut, MixUp, CutMix), cosine annealing LR scheduling, progressive unfreezing, INT8 model quantization, Grad-CAM interpretability, CLI inference tools, and a Streamlit demo app.

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

## Motivation

> _How much does a pretrained backbone actually help compared to training from scratch — when both models share the same training budget?_

This project answers that question through a **controlled experiment**: identical dataset, optimizer, learning rate, and epoch count — the only variable is the architecture and whether the weights are pretrained. The results have direct implications for real-world model selection and deployment strategy.

---

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

| Epoch | Custom CNN (Val Acc) | MobileNetV2 (Val Acc) |
|:-----:|:--------------------:|:---------------------:|
| 1 | 27.1% | **83.0%** |
| 2 | 36.8% | 84.6% |
| 3 | 44.8% | 85.4% |
| 4 | 49.1% | 85.5% |
| 5 | 56.0% | **85.5%** |

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

### MobileNetV2 — Transfer Learning (Frozen Backbone)

```
Pretrained MobileNetV2 (ImageNet weights — FROZEN)
  |
  +-- Classifier Head: Dropout(0.2) -> Linear(1280 -> 10)   <-- only trainable layer
```

**Strategy:** Freeze the entire feature extraction backbone (2.2M params) and train only a lightweight classifier head (12,810 params). Input images are resized from 32x32 to 224x224 to match the expected ImageNet resolution.

---

## Error Analysis

Both models consistently confuse visually similar classes — but MobileNetV2 makes significantly fewer mistakes:

| Confusion Pair | Custom CNN Errors | MobileNetV2 Errors | Reduction | Root Cause |
|:--------------|:-----------------:|:-------------------:|:---------:|:-----------|
| Cat <-> Dog | 431 | 177 | **59%** | Similar fur texture and body structure |
| Horse <-> Dog | 231 | 75 | **68%** | Quadruped body shape overlap |
| Ship <-> Airplane | 226 | 55 | **76%** | Shared sky/water backgrounds |
| Truck <-> Automobile | 220 | 53 | **76%** | Rectangular vehicle shapes at 32x32 |
| Bird <-> Deer | 219 | 49 | **78%** | Challenging at low resolution |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

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

> The CIFAR-10 dataset is downloaded automatically on first run via `torchvision.datasets`. GPU (CUDA) and Apple Silicon (MPS) acceleration are auto-detected.

---

## Usage

### Inference (CLI)

```bash
# Classify random CIFAR-10 test images
python predict.py --test-samples 10 --model both

# Classify a single image
python predict.py --image path/to/image.png --model mobilenet

# Classify a directory of images
python predict.py --image-dir path/to/images/ --model both --save results/predictions.png
```

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

```
CIFAR-10-Image-Classification/
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

---

## Key Takeaways

1. **Transfer learning is remarkably efficient** — A frozen MobileNetV2 backbone with a linear classifier (12K trainable params) outperforms a fully-trained 2.4M-parameter CNN by ~30 percentage points.

2. **Pretrained features generalize across domains** — Despite the resolution gap (ImageNet 224x224 vs. CIFAR-10 32x32), learned representations transfer effectively with proper resizing.

3. **More parameters ≠ better performance** — The Custom CNN has 200x more trainable parameters yet achieves significantly lower accuracy under the same training budget.

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

**Built with PyTorch** · Designed for Learning & Demonstration

</div>
