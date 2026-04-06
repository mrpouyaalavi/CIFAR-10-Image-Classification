<div align="center">

<!-- Typing animation -->
[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2800&pause=700&color=EE4C2C&center=true&vCenter=true&width=860&lines=CIFAR-10+Deep+Learning+Image+Classification;Custom+CNN+vs+Transfer+Learning+%E2%80%94+A+Comparative+Study;PyTorch+%C2%B7+Grad-CAM+%C2%B7+MobileNetV2+%C2%B7+Streamlit;5+Architectures+%C2%B7+85.53%25+Accuracy+%C2%B7+Full+ML+Pipeline)](https://readme-typing-svg.demolab.com)

<!-- Badges -->
![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

# CIFAR-10 Image Classification — Deep Learning from Scratch to Production

> **An end-to-end deep learning pipeline that designs, trains, and rigorously evaluates five architectures on the CIFAR-10 benchmark — proving that transfer learning with 200× fewer trainable parameters outperforms a fully-trained custom CNN by 30 percentage points.**

This project is a comprehensive, portfolio-grade machine learning study that goes beyond model training. It includes data augmentation pipelines (RandomCrop, CutOut, MixUp, CutMix), cosine annealing LR scheduling, progressive unfreezing, INT8 model quantization, Grad-CAM interpretability visualizations, a CLI inference toolkit, and a Streamlit demo app — all documented across a 14-section Jupyter Notebook.

**[📓 Explore the Notebook](cifar10%20image%20classification.ipynb)** &nbsp;·&nbsp; **[🚀 Run the Demo](#-streamlit-demo-app)** &nbsp;·&nbsp; **[📊 Key Results](#-key-results--performance-benchmarks)**

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🎯 Motivation & Research Question

> _How much does a pretrained backbone actually help compared to training from scratch — when both models share the same training budget?_

Deep learning practitioners often default to transfer learning without quantifying its advantage under controlled conditions. This project answers that question through a **rigorous, controlled experiment**: identical dataset, optimizer, learning rate, epoch count, and augmentation strategy — the only variable is the architecture and whether the weights are pretrained.

The results have direct implications for:

- **Model selection** in resource-constrained environments (edge, mobile, embedded)
- **Training efficiency** when labelled data is limited
- **Deployment strategy** when balancing latency versus accuracy

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📊 Key Results & Performance Benchmarks

<div align="center">

| Metric | Custom CNN | MobileNetV2 | Winner |
|:-------|:----------:|:-----------:|:------:|
| **Test Accuracy** | 55.96% | **85.53%** | 🏆 MobileNetV2 |
| **Trainable Params** | 2,462,282 | **12,810** | 🏆 MobileNetV2 |
| **Model Size** | 9.40 MB | **8.66 MB** | 🏆 MobileNetV2 |
| **Inference Latency** | **0.84 ms** | 5.33 ms | 🏆 Custom CNN |
| **Throughput** | **~1,191 FPS** | ~188 FPS | 🏆 Custom CNN |

</div>

> **Key Finding:** MobileNetV2 achieves **85.53% accuracy** with just **0.5%** of the Custom CNN's trainable parameters. Transfer learning doesn't just win on accuracy — it wins with **200× fewer trainable weights**.

### Training Progression — Convergence Comparison

```
Epoch   Custom CNN (Val Acc)     MobileNetV2 (Val Acc)
─────   ────────────────────     ─────────────────────
  1          27.1%                    83.0%  ◀ Already 83% after ONE epoch
  2          36.8%                    84.6%
  3          44.8%                    85.4%
  4          49.1%                    85.5%
  5          56.0%                    85.5%  ◀ Converged
```

MobileNetV2 reaches **83% after a single epoch**. The Custom CNN is still at 27% — demonstrating the enormous advantage of pretrained feature representations even when input resolution is drastically different (ImageNet 224×224 vs CIFAR-10 32×32).

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🧠 Model Architectures

### Custom CNN — 4-Block Design (Trained From Scratch)

A purpose-built convolutional network with progressive channel expansion, dual convolutions per block, and aggressive regularization:

```
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
- **Kaiming He initialization** for stable gradient flow through deep ReLU networks
- **Dual convolutions per block** to increase receptive field before downsampling
- **Global Average Pooling** (Block 4) eliminates large FC layers, reducing overfitting
- **Aggressive dropout** (0.25 in conv blocks, 0.5 in classifier) for regularization

### MobileNetV2 — Transfer Learning (Frozen ImageNet Backbone)

```
Pretrained MobileNetV2 (ImageNet — 1.2M images, 1000 classes — FROZEN)
  │
  └── Classifier Head: Dropout(0.2) → Linear(1280 → 10)   ◀ Only trainable layer
```

**Strategy:** Freeze the entire feature extraction backbone (2.2M params) and train only a lightweight classifier head (12,810 params). Depthwise separable convolutions reduce computation ~8–9× compared to standard convolutions, making it practical for edge deployment.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 🔍 Error Analysis & Confusion Patterns

Both models consistently confuse visually similar classes — but MobileNetV2 makes significantly fewer mistakes:

<div align="center">

| Confusion Pair | Custom CNN Errors | MobileNetV2 Errors | Error Reduction | Root Cause |
|:--------------|:-----------------:|:-------------------:|:---------------:|:-----------|
| 🐱 Cat ↔ 🐕 Dog | 431 | 177 | **59%** | Similar fur texture and body structure |
| 🐴 Horse ↔ 🐕 Dog | 231 | 75 | **68%** | Quadruped body shape overlap |
| 🚢 Ship ↔ ✈️ Airplane | 226 | 55 | **76%** | Shared sky/water backgrounds |
| 🚚 Truck ↔ 🚗 Automobile | 220 | 53 | **76%** | Rectangular vehicle shapes at 32×32 |
| 🐦 Bird ↔ 🦌 Deer | 219 | 49 | **78%** | Challenging at low resolution |

</div>

> The pretrained ImageNet features give MobileNetV2 a decisive advantage in disambiguating fine-grained visual differences that a model trained from scratch on 32×32 images struggles to capture.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## Key Features

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  🧠  5 architectures benchmarked in notebook (CLI tools: CNN & MobileNetV2) ║
║  📈  Full training pipeline with cosine annealing LR & progressive unfreezing║
║  🎲  Advanced augmentation: RandomCrop, CutOut, MixUp, CutMix               ║
║  🔬  Grad-CAM interpretability — see what the model actually looks at        ║
║  ⚡  INT8 dynamic quantization for deployment-ready performance              ║
║  📊  Confusion matrices, training curves, and efficiency benchmarks          ║
║  🖥️  Streamlit demo app — interactive side-by-side model comparison          ║
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
| **Demo App** | Streamlit |
| **Environment** | Jupyter Notebook, Python 3.11 |
| **Hardware** | Auto-detected: CUDA / Apple Silicon MPS / CPU |

### Training Hyperparameters

Both models were trained with **identical hyperparameters** — the only variable is the architecture:

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

> The CIFAR-10 dataset is downloaded automatically on first run via `torchvision.datasets`. GPU (CUDA) and Apple Silicon (MPS) acceleration are auto-detected.

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

### Grad-CAM Visualizations (CLI)

```bash
# Generate Grad-CAM heatmaps for both models
python gradcam.py --model both --num-images 6

# Visualize specific test images and save output
python gradcam.py --model both --image-index 0 42 100 --save results/gradcam/
```

### 🖥️ Streamlit Demo App

```bash
streamlit run app.py
```

Upload any image or sample from the CIFAR-10 test set for **interactive side-by-side model comparison** with confidence progress bars, top-k predictions, and device info.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 📁 Project Structure

```text
CIFAR-10-Image-Classification/
│
├── cifar10 image classification.ipynb   # Main notebook — full 14-section ML pipeline
├── predict.py                            # CLI inference — single / batch / directory
├── gradcam.py                            # Grad-CAM interpretability visualizations
├── app.py                                # Streamlit demo — interactive classification
│
├── artifacts/
│   └── run_config.json                   # Training hyperparameters & experiment config
├── results/
│   └── training_metadata.json            # Experiment results, metrics & training history
│
├── checkpoints/                          # Model checkpoints (auto-generated)
│   ├── custom_cnn_best.pth
│   ├── mobilenetv2_best.pth
│   └── ...
├── models/                               # Saved model weights (auto-generated)
│   ├── custom_cnn_model.pth
│   ├── mobilenet_model.pth
│   └── ...
├── data/                                 # CIFAR-10 dataset (auto-downloaded)
│
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── .gitignore                            # Git ignore rules
```

> Model weights (`.pth`), datasets (`data/`), and checkpoints (`checkpoints/`) are excluded via `.gitignore` and regenerated automatically when running the notebook.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## 💡 Key Takeaways

1. **Transfer learning is remarkably efficient** — A frozen MobileNetV2 backbone with a linear classifier (12K trainable params) outperforms a fully-trained 2.4M-parameter CNN by ~30 percentage points.

2. **Pretrained features generalize across domains** — Despite the resolution gap (ImageNet 224×224 vs. CIFAR-10 32×32), learned representations transfer effectively with proper resizing.

3. **More parameters ≠ better performance** — The Custom CNN has 200× more trainable parameters yet achieves significantly lower accuracy under the same training budget.

4. **Data efficiency matters** — With limited training data, transfer learning reaches production-grade accuracy while training from scratch barely starts to converge.

5. **Speed vs. accuracy trade-off** — The Custom CNN is 6× faster at inference (0.84ms vs 5.33ms) due to native 32×32 input — relevant for latency-critical edge deployments.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f172a,30:EE4C2C,60:F7931E,100:0f172a&height=2" width="100%"/>

<br/>

## ✅ Roadmap

- [x] Train with the **full CIFAR-10 dataset** (50K images) with cosine annealing LR
- [x] Implement **advanced data augmentation** (RandomCrop, CutOut, MixUp, CutMix)
- [x] Benchmark **5 architectures** (Custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0, ViT)
- [x] **Progressive unfreezing** — 3-phase MobileNetV2 fine-tuning strategy
- [x] **INT8 model quantization** for deployment-ready performance
- [x] **Grad-CAM interpretability** — visualize model attention regions
- [x] **CLI inference tools** — single image, batch, and test-sample modes
- [x] **Streamlit demo app** — interactive side-by-side model comparison

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

```
> Target     : Pouya Alavi Naeini — AI & Full-Stack Developer
> University : Macquarie University, Sydney, NSW
> Major      : B.IT — Artificial Intelligence & Web/App Development
> Status     : [●] ONLINE — open to grad & junior opportunities
```

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-EE4C2C?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0f172a)](https://www.linkedin.com/in/pouya-alavi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-F7931E?style=for-the-badge&logo=github&logoColor=ffffff&labelColor=0f172a)](https://github.com/mrpouyaalavi)
[![Email](https://img.shields.io/badge/Email-Contact-f59e0b?style=for-the-badge&logo=gmail&logoColor=09090b&labelColor=0f172a)](mailto:pouyaalavi1378@gmail.com)

<br/>

**Built with PyTorch** · Designed for Learning, Research & Demonstration

</div>
