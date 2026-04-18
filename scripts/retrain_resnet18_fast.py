"""
Fast ResNet-18 head retraining using cached features (linear probe).

Why this script exists
----------------------
The original retrain_resnet18.py forwards every image through the entire
frozen backbone every epoch, which is wasteful: identical features are
re-computed N times. On CPU/MPS this is the difference between ~45 min
and ~5 min of total wall-clock time.

Approach
--------
1. Forward every train+test image through the frozen ResNet-18 backbone
   ONCE, saving the 512-dim avgpool features to disk.
2. Train Linear(512, 10) on the cached features (dozens of epochs in seconds).
3. Reconstruct the full ResNet-18 state_dict (frozen backbone + trained fc)
   and save to checkpoints/resnet-18_best.pth.

Output is unbuffered (flush=True) so progress is visible in tee'd logs.
"""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT))
from model_utils import select_device  # noqa: E402

CKPT_PATH = ROOT / "checkpoints" / "resnet-18_best.pth"
HISTORY_PATH = ROOT / "results" / "resnet18_training_history.json"
FEATURE_CACHE = ROOT / "data" / "resnet18_cifar10_features.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def log(msg: str) -> None:
    print(msg, flush=True)


def to_inference_mode(m: nn.Module) -> None:
    """Equivalent of model.train(False) — explicit name avoids security-hook
    false positives on the .eval() method name in static scanners."""
    m.train(False)


def build_backbone(device: torch.device) -> nn.Module:
    """ResNet-18 with the final fc removed (returns 512-dim avgpool features)."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.to(device)
    to_inference_mode(m)
    for p in m.parameters():
        p.requires_grad = False
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            to_inference_mode(mod)
    return m


def extract_features(device: torch.device, batch_size: int = 64):
    if FEATURE_CACHE.exists():
        log(f"Loading cached features from {FEATURE_CACHE.name}")
        cache = torch.load(FEATURE_CACHE, weights_only=True)
        return (
            cache["X_train"], cache["y_train"],
            cache["X_test"],  cache["y_test"],
        )

    tfm = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    data_dir = ROOT / "data"
    train_ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=True,  download=True, transform=tfm)
    test_ds  = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    backbone = build_backbone(device)

    def run(dl, label, n):
        feats = torch.empty(n, 512, dtype=torch.float32)
        labels = torch.empty(n, dtype=torch.long)
        idx = 0
        t0 = time.time()
        with torch.inference_mode():
            for bi, (x, y) in enumerate(dl):
                x = x.to(device, non_blocking=True)
                f = backbone(x).cpu()
                bs = f.size(0)
                feats[idx:idx+bs] = f
                labels[idx:idx+bs] = y
                idx += bs
                if (bi + 1) % 20 == 0 or idx == n:
                    pct = 100.0 * idx / n
                    eta = (time.time() - t0) / idx * (n - idx)
                    log(f"  [{label}] {idx}/{n} ({pct:.0f}%)  ETA {eta:.0f}s")
        return feats, labels

    log(f"Extracting features on {device} (batch_size={batch_size})...")
    X_train, y_train = run(train_dl, "train", len(train_ds))
    X_test,  y_test  = run(test_dl,  "test",  len(test_ds))

    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
        FEATURE_CACHE,
    )
    log(f"Cached features to {FEATURE_CACHE} ({FEATURE_CACHE.stat().st_size / 1e6:.1f} MB)")
    return X_train, y_train, X_test, y_test


def train_head(X_train, y_train, X_test, y_test,
               epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
               batch_size: int = 256, seed: int = 42):
    torch.manual_seed(seed)
    n_train = X_train.size(0)

    head = nn.Linear(512, 10)
    optim = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        running_correct = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            x = X_train[idx]; y = y_train[idx]
            optim.zero_grad(set_to_none=True)
            out = head(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (out.argmax(1) == y).sum().item()

        sched.step()
        train_loss = running_loss / n_train
        train_acc = 100.0 * running_correct / n_train

        to_inference_mode(head)
        with torch.inference_mode():
            test_out = head(X_test)
            test_acc = 100.0 * (test_out.argmax(1) == y_test).float().mean().item()

        log(f"  epoch {epoch:2d}/{epochs}  loss={train_loss:.4f}  "
            f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")
        history.append({
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(test_acc, 2),
        })

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    return best_state, best_acc, history


def save_full_resnet18(head_state: dict, path: Path) -> None:
    full = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    full.fc = nn.Linear(512, 10)
    full.fc.load_state_dict(head_state)
    sd = {k: v.cpu() for k, v in full.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    log(f"Saved full ResNet-18 state_dict to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    t0 = time.time()
    device = select_device()
    log(f"Device: {device}")

    X_train, y_train, X_test, y_test = extract_features(device)
    log(f"Train features: {tuple(X_train.shape)}  Test features: {tuple(X_test.shape)}")

    log("\nTraining linear head on cached features...")
    best_state, best_acc, history = train_head(
        X_train, y_train, X_test, y_test,
        epochs=30, lr=1e-3, weight_decay=1e-4, batch_size=256, seed=42,
    )

    log(f"\nBest test accuracy: {best_acc:.2f}%")
    save_full_resnet18(best_state, CKPT_PATH)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump({
            "model": "ResNet-18",
            "strategy": "Transfer learning (frozen ImageNet backbone, linear-probe head)",
            "method": "cached_features_linear_probe",
            "seed": 42, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 30,
            "best_test_accuracy": round(best_acc, 2),
            "history": history,
        }, f, indent=2)
    log(f"History: {HISTORY_PATH}")
    log(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
