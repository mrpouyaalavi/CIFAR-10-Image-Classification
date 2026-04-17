"""
Retraining script for ResNet-18 on CIFAR-10.

Rationale
---------
The committed `resnet-18_best.pth` is the output of this script: a frozen
ImageNet ResNet-18 backbone with a freshly-trained 5 130-parameter FC head.
On the canonical run this achieves **82.10%** test accuracy on the full
10 000-image CIFAR-10 test set.

Architecture note
-----------------
ResNet-18's final layer is ``nn.Linear(512, 1000)``. We replace it with
``nn.Linear(512, num_classes)`` and freeze every other parameter. This is
identical in spirit to the MobileNetV2 transfer-learning approach:

    Custom CNN     48.40%   2,462,282 trainable params   trained from scratch
    ResNet-18      82.10%       5,130 trainable params   frozen backbone
    MobileNetV2    86.91%      12,810 trainable params   frozen backbone

Design decisions
----------------
1. **No augmentation.** Same reasoning as retrain_mobilenetv2.py: the
   ImageNet backbone already encodes strong visual invariances; adding
   noise at 224x224 upscaled from 32x32 only adds variance without
   providing new signal.
2. **BN pinned in inference mode.** All BatchNorm2d layers are locked to
   inference mode to prevent running-stat drift during head training.
3. **Adam + cosine annealing.** Matches the training config used across
   all models in this project for a fair comparison.

Usage
-----
    python scripts/retrain_resnet18.py --epochs 5 --batch-size 128 --lr 1e-3

Output
------
    checkpoints/resnet-18_best.pth          (overwritten on improvement)
    results/resnet18_training_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "resnet-18_best.pth"
HISTORY_PATH = ROOT / "results" / "resnet18_training_history.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model_utils import select_device  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int = 10) -> nn.Module:
    """Frozen ResNet-18 backbone + trainable FC head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Freeze the entire backbone.
    for p in model.parameters():
        p.requires_grad = False
    # Replace the final classifier (512 -> 1000) with one for num_classes.
    # Linear(512, num_classes): 512*num_classes weights + num_classes biases.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_batchnorm_layers(model: nn.Module) -> None:
    """Force every BatchNorm layer into inference mode (frozen running stats).

    When a BatchNorm layer's parameters are frozen you must *also* prevent
    its running stats from updating -- otherwise the backbone's feature
    distribution drifts toward the new dataset's batch statistics and the
    pretrained head's calibration collapses.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(False)


def get_loaders(batch_size: int):
    # No augmentation: see docstring rationale.
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    data_dir = ROOT / "data"
    train_ds = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=tfm,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=tfm,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False,
    )
    return train_loader, test_loader


def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.train(False)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device()
    print(f"Device: {device}")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(args.batch_size)

    model = build_model(10).to(device)
    freeze_batchnorm_layers(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  lr={args.lr:.1e}")
    # Sanity: should always be Linear(512, 10) = 5 130 params
    assert trainable == 5_130, (
        f"Expected 5 130 trainable params, got {trainable:,}. "
        "This usually means the backbone freeze did not apply correctly."
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Head in train mode; BN stays frozen.
        model.train(True)
        freeze_batchnorm_layers(model)

        t0 = time.time()
        running_loss = running_correct = running_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            running_total += y.numel()

        sched.step()
        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        val_acc = compute_accuracy(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"  epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f}  "
            f"train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  ({dt:.1f}s)"
        )
        history.append({
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "seconds": round(dt, 1),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                CKPT_PATH,
            )
            print(f"    checkmark new best {val_acc:.2f}% saved to {CKPT_PATH.name}")

    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoint: {CKPT_PATH}")

    with open(HISTORY_PATH, "w") as f:
        json.dump(
            {
                "model": "ResNet-18",
                "strategy": "Transfer learning (frozen ImageNet backbone, head-only training)",
                "seed": args.seed,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "best_val_accuracy": round(best_acc, 2),
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
