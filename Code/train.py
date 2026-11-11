"""
train.py - PyTorch training script for RealMe deepfake detector using EfficientNet-B0.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root and Code directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(pretrained: bool = True) -> nn.Module:
    """Creates an EfficientNet-B0 classifier with a single output logit."""
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)
    return model


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates model performance and computes standard classification metrics."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_targets.extend(targets.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    total_samples = len(all_targets)
    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": auc,
    }


def run_smoke_test(epochs: int = 1):
    """Creates a tiny synthetic dataset in memory to verify training and pipeline."""
    print("\n--- Running RealMe Smoke Test with Synthetic In-Memory Data ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy images: (N, 3, 224, 224) and labels: 0.0 (real) or 1.0 (fake)
    n_train, n_val, n_test = 20, 8, 8
    x_train = torch.randn(n_train, 3, 224, 224)
    y_train = torch.tensor([0.0, 1.0] * (n_train // 2), dtype=torch.float32)

    x_val = torch.randn(n_val, 3, 224, 224)
    y_val = torch.tensor([0.0, 1.0] * (n_val // 2), dtype=torch.float32)

    x_test = torch.randn(n_test, 3, 224, 224)
    y_test = torch.tensor([0.0, 1.0] * (n_test // 2), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=4, shuffle=False)

    model = create_model(pretrained=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\n================== TEST RESULTS ==================")
    print(f"Accuracy : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-Score : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print("==================================================")

    # Save test weights
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Smoke test weights successfully saved to: {save_path}\n")


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    data_path = Path(args.data)
    if not data_path.exists() or args.smoke_test:
        if not data_path.exists():
            print(f"Data directory '{data_path}' not found.")
        run_smoke_test(epochs=args.epochs)
        return

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_path,
        batch_size=args.batch_size,
        subset=args.subset,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    model = create_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = -1.0
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch} Summary: Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        # Save best model checkpoint
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Saved new best model to {best_model_path} (Val AUC: {best_val_auc:.4f})")

    # Final evaluation on Test split
    print("\nRunning final evaluation on test set...")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\n================== TEST RESULTS ==================")
    print(f"Accuracy : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-Score : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealMe Deepfake Detection Training")
    parser.add_argument("--data", type=str, default="data/real_vs_fake/real-vs-fake", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--subset", type=int, default=None, help="Sample subset size for quick test")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run in-memory smoke test")

    args = parser.parse_args()
    train(args)
