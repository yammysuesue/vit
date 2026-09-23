"""
Stage 1: Train a classification head on frozen DINOv2 ViT-B/14 features.

Head options:
  linear     – CLS token → Linear(768, C)
  mlp        – CLS token → 2-layer MLP
  attn_pool  – CLS token ⊕ attention-pooled patch tokens → MLP  (recommended)

Usage:
    conda activate dp310
    python train_dinov2_stage1.py --head attn_pool --batch_size 64 --epochs 50
"""

import argparse
import os
import sys
import time
import copy
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dinov2"))
from dinov2.models.vision_transformer import vit_base


def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 Stage-1: frozen backbone + head")
    p.add_argument("--data_dir", type=str, default="data/train_val")
    p.add_argument("--test_dir", type=str, default="data/test")
    p.add_argument("--weight_path", type=str, default="weight/dinov2_vitb14_pretrain.pth")
    p.add_argument("--head", choices=["linear", "mlp", "attn_pool"], default="attn_pool")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="output_stage1")
    p.add_argument("--img_size", type=int, default=224)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backbone(weight_path: str, device: torch.device) -> nn.Module:
    backbone = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-5)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


class AttentionPoolingHead(nn.Module):
    """
    Learnable attention pooling over patch tokens, concatenated with the CLS
    token, followed by an MLP classifier.

    Architecture:
        CLS token (D) ──────────────────────────┐
                                                 │  concat
        Patch tokens (N×D) → AttnPool → (D) ────┘
                                                 │
                                            LayerNorm(2D)
                                            Linear(2D → hidden)
                                            GELU
                                            Dropout
                                            Linear(hidden → C)

    The attention pool uses a single learnable query that attends to all N
    patch tokens via scaled dot-product attention, allowing the model to
    learn *which spatial regions* are most discriminative for weed species.
    """

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = cls_token.shape[0]
        q = self.query.expand(B, -1, -1)                       # (B, 1, D)
        pooled, _ = self.attn(q, patch_tokens, patch_tokens)    # (B, 1, D)
        pooled = self.attn_norm(pooled.squeeze(1))              # (B, D)
        fused = torch.cat([cls_token, pooled], dim=1)           # (B, 2D)
        return self.classifier(fused)


def build_head(head_type: str, embed_dim: int, num_classes: int) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(embed_dim, num_classes)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    if head_type == "attn_pool":
        return AttentionPoolingHead(embed_dim, num_classes)
    raise ValueError(f"Unknown head type: {head_type}")


def build_transforms(img_size: int):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def stratified_split(dataset, val_ratio: float, seed: int):
    targets = np.array([s[1] for s in dataset.samples])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(targets, targets))
    return train_idx, val_idx


class TransformSubset(torch.utils.data.Dataset):
    """Subset of an ImageFolder with its own transform."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


def compute_class_weights(dataset, indices, num_classes, device):
    targets = np.array([dataset.samples[i][1] for i in indices])
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights).to(device)


def _forward_head(head, backbone_out):
    """Dispatch forward through the head, handling both simple and attn_pool heads."""
    if isinstance(head, AttentionPoolingHead):
        return head(backbone_out["x_norm_clstoken"], backbone_out["x_norm_patchtokens"])
    return head(backbone_out["x_norm_clstoken"])


def train_one_epoch(backbone, head, loader, criterion, optimizer, device):
    head.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.no_grad():
            features = backbone.forward_features(imgs)

        logits = _forward_head(head, features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(backbone, head, loader, criterion, device):
    head.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        features = backbone.forward_features(imgs)
        logits = _forward_head(head, features)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return running_loss / total, running_correct / total, all_preds, all_labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────
    train_tf, val_tf = build_transforms(args.img_size)
    full_dataset = datasets.ImageFolder(args.data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    train_idx, val_idx = stratified_split(full_dataset, args.val_ratio, args.seed)
    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples = [full_dataset.samples[i] for i in val_idx]
    train_set = TransformSubset(train_samples, train_tf)
    val_set = TransformSubset(val_samples, val_tf)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    g = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, generator=g)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # ── model ─────────────────────────────────────────────────────────
    backbone = build_backbone(args.weight_path, device)
    embed_dim = backbone.embed_dim  # 768
    head = build_head(args.head, embed_dim, num_classes).to(device)

    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Head type: {args.head}  |  Trainable params: {trainable:,}")

    # ── optimiser / loss ──────────────────────────────────────────────
    class_weights = compute_class_weights(full_dataset, train_idx, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── training loop ─────────────────────────────────────────────────
    best_val_acc = 0.0
    best_head_state = None
    patience_counter = 0

    log_path = os.path.join(args.save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(backbone, head, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(backbone, head, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} {train_loss:11.4f} {train_acc:10.4f} {val_loss:9.4f} {val_acc:8.4f} {lr_now:10.6f}  ({elapsed:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{val_loss:.5f}", f"{val_acc:.5f}", f"{lr_now:.7f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
            torch.save(best_head_state, os.path.join(args.save_dir, "best_head.pth"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # ── test evaluation ───────────────────────────────────────────────
    if os.path.isdir(args.test_dir):
        print("\n" + "=" * 62)
        print("Evaluating on competition test set ...")
        test_dataset = datasets.ImageFolder(args.test_dir, transform=val_tf)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        test_class_names = test_dataset.classes

        head.load_state_dict(best_head_state)

        # Map test folder names to train class indices
        # Test folders have "-compet" suffix, e.g. "Carpetweed-compet" -> "Carpetweed"
        test_to_train = {}
        for tidx, tname in enumerate(test_class_names):
            base = tname.replace("-compet", "")
            if base in class_names:
                test_to_train[tidx] = class_names.index(base)
            else:
                print(f"  WARNING: test class '{tname}' has no train match")

        head.eval()
        correct, total = 0, 0
        all_preds, all_labels, all_mapped = [], [], []
        total_inference_time = 0.0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=True)

                t0 = time.time()
                features = backbone.forward_features(imgs)
                logits = _forward_head(head, features)
                total_inference_time += time.time() - t0

                preds = logits.argmax(1).cpu()
                for pred, label in zip(preds, labels):
                    mapped_label = test_to_train.get(label.item())
                    if mapped_label is not None:
                        all_preds.append(pred.item())
                        all_mapped.append(mapped_label)
                        if pred.item() == mapped_label:
                            correct += 1
                        total += 1

        test_acc = correct / total if total > 0 else 0
        avg_time_ms = (total_inference_time / total) * 1000 if total > 0 else 0

        print(f"\nTest accuracy: {test_acc:.4f} ({correct}/{total})")
        print(f"Avg inference time per image: {avg_time_ms:.2f} ms")

        mapped_names = class_names
        print(f"\nPer-class report:\n{classification_report(all_mapped, all_preds, target_names=mapped_names, zero_division=0)}")

        cm = confusion_matrix(all_mapped, all_preds)
        print("Confusion matrix:")
        print(cm)

        with open(os.path.join(args.save_dir, "test_results.txt"), "w") as f:
            f.write(f"Test accuracy: {test_acc:.4f} ({correct}/{total})\n")
            f.write(f"Avg inference time: {avg_time_ms:.2f} ms/image\n\n")
            f.write(classification_report(all_mapped, all_preds, target_names=mapped_names, zero_division=0))
    else:
        print(f"\nTest directory '{args.test_dir}' not found — skipping test eval.")

    # ── save final artefacts ──────────────────────────────────────────
    save_bundle = {
        "head_state_dict": best_head_state,
        "head_type": args.head,
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    torch.save(save_bundle, os.path.join(args.save_dir, "stage1_bundle.pth"))
    print(f"\nAll artefacts saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
