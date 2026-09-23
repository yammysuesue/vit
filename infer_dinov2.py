"""
Inference script for DINOv2 weed classifier (Stage 1).

Supports three input modes:
  1. Single image:       python infer_dinov2.py --input path/to/image.jpg
  2. Directory of images: python infer_dinov2.py --input path/to/folder/
  3. Test set with GT:   python infer_dinov2.py --input data/test --has_labels

Usage:
    conda activate dp310
    python infer_dinov2.py --input data/test --has_labels
    python infer_dinov2.py --input data/test/Carpetweed-compet/Carpetweeds_Mod_20.jpg
    python infer_dinov2.py --input some_folder_of_images/
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dinov2"))
from dinov2.models.vision_transformer import vit_base

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Head architectures (must match train_dinov2_stage1.py) ────────────

class AttentionPoolingHead(nn.Module):
    """CLS token + attention-pooled patch tokens → MLP classifier."""

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
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, patch_tokens, patch_tokens)
        pooled = self.attn_norm(pooled.squeeze(1))
        fused = torch.cat([cls_token, pooled], dim=1)
        return self.classifier(fused)


def _build_head(head_type: str, embed_dim: int, num_classes: int) -> nn.Module:
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


def _forward_head(head, backbone_out):
    if isinstance(head, AttentionPoolingHead):
        return head(backbone_out["x_norm_clstoken"], backbone_out["x_norm_patchtokens"])
    return head(backbone_out["x_norm_clstoken"])


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 weed classifier — inference")
    p.add_argument("--input", type=str, required=True,
                   help="Path to a single image, a flat directory of images, or an ImageFolder-style directory")
    p.add_argument("--bundle", type=str, default="output_stage1/stage1_bundle.pth",
                   help="Path to the saved stage1_bundle.pth")
    p.add_argument("--backbone_weights", type=str, default="weight/dinov2_vitb14_pretrain.pth",
                   help="Path to DINOv2 ViT-B/14 pretrained weights")
    p.add_argument("--has_labels", action="store_true",
                   help="If set, treat --input as an ImageFolder with ground-truth subfolders")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_csv", type=str, default=None,
                   help="Optional path to save predictions as CSV")
    p.add_argument("--save_cm", type=str, default=None,
                   help="Optional path to save the confusion matrix as CSV when --has_labels is set")
    p.add_argument("--save_error_dir", type=str, default=None,
                   help="Optional directory to save misclassification analysis when --has_labels is set")
    p.add_argument("--max_error_examples", type=int, default=12,
                   help="Maximum number of misclassified examples to visualize")
    return p.parse_args()


def build_transform(img_size: int):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ImageListDataset(Dataset):
    """Dataset from a flat list of image file paths (no labels)."""
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


def _draw_text_block(draw, xy, lines, font, fill=(17, 24, 39), line_gap=4):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_gap


def _load_thumb(image_path: str, size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    thumb = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    return thumb


def _find_support_example(samples, class_to_idx, class_name, exclude_path=None):
    class_idx = class_to_idx.get(class_name)
    if class_idx is None:
        return None
    for sample_path, sample_label in samples:
        if sample_label == class_idx and sample_path != exclude_path:
            return sample_path
    return None


def _get_error_dir(save_error_dir, save_csv):
    if save_error_dir:
        return Path(save_error_dir)
    if save_csv:
        return Path(save_csv).with_name(f"{Path(save_csv).stem}_error_analysis")
    return Path.cwd() / "error_analysis"


def save_misclassification_analysis(misclassified, test_dataset, error_dir, max_error_examples):
    error_dir.mkdir(parents=True, exist_ok=True)

    summary_txt_path = error_dir / "misclassification_summary.txt"
    summary_csv_path = error_dir / "misclassification_details.csv"
    collage_path = error_dir / "misclassified_grid.png"
    pair_dir = error_dir / "pair_analysis"
    pair_dir.mkdir(parents=True, exist_ok=True)

    pair_counts = Counter((item["ground_truth"], item["prediction"]) for item in misclassified)
    class_error_counts = Counter(item["ground_truth"] for item in misclassified)
    avg_conf_by_pair = {}
    for pair in pair_counts:
        confidences = [item["confidence"] for item in misclassified if (item["ground_truth"], item["prediction"]) == pair]
        avg_conf_by_pair[pair] = sum(confidences) / len(confidences)

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true_class", "predicted_class", "confidence"])
        for item in misclassified:
            writer.writerow([os.path.basename(item["path"]), item["ground_truth"], item["prediction"], f"{item['confidence']:.4f}"])

    with open(summary_txt_path, "w") as f:
        f.write(f"Total misclassified samples: {len(misclassified)}\n\n")
        f.write("Errors by true class:\n")
        for class_name, count in class_error_counts.most_common():
            f.write(f"  - {class_name}: {count}\n")
        f.write("\nMost common confusion pairs:\n")
        for (true_name, pred_name), count in pair_counts.most_common():
            f.write(f"  - {true_name} -> {pred_name}: {count} (avg conf {avg_conf_by_pair[(true_name, pred_name)]:.4f})\n")
        f.write("\nLowest-confidence errors:\n")
        for item in sorted(misclassified, key=lambda x: x["confidence"])[: min(5, len(misclassified))]:
            f.write(
                f"  - {os.path.basename(item['path'])}: true={item['ground_truth']}, "
                f"pred={item['prediction']}, conf={item['confidence']:.4f}\n"
            )

    font = ImageFont.load_default()

    selected = sorted(misclassified, key=lambda x: x["confidence"])[: max_error_examples]
    if selected:
        cols = min(3, len(selected))
        rows = (len(selected) + cols - 1) // cols
        tile_w, tile_h = 300, 340
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(248, 250, 252))
        for idx, item in enumerate(selected):
            col = idx % cols
            row = idx // cols
            x0 = col * tile_w
            y0 = row * tile_h
            thumb = _load_thumb(item["path"], size=(260, 220))
            canvas.paste(thumb, (x0 + 20, y0 + 16))
            draw = ImageDraw.Draw(canvas)
            _draw_text_block(
                draw,
                (x0 + 20, y0 + 250),
                [
                    os.path.basename(item["path"]),
                    f"True: {item['ground_truth']}",
                    f"Pred: {item['prediction']}",
                    f"Conf: {item['confidence']:.4f}",
                ],
                font,
            )
        canvas.save(collage_path)

    grouped = defaultdict(list)
    for item in misclassified:
        grouped[(item["ground_truth"], item["prediction"])].append(item)

    for pair_rank, ((true_name, pred_name), pair_count) in enumerate(pair_counts.most_common(5), start=1):
        pair_items = grouped[(true_name, pred_name)]
        example = sorted(pair_items, key=lambda x: x["confidence"])[0]
        true_ref = _find_support_example(test_dataset.samples, test_dataset.class_to_idx, true_name, exclude_path=example["path"])
        pred_ref = _find_support_example(test_dataset.samples, test_dataset.class_to_idx, pred_name)

        panel = Image.new("RGB", (980, 330), color=(255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 979, 329), outline=(203, 213, 225), width=2)
        draw.text((22, 16), f"Confusion #{pair_rank}: {true_name} -> {pred_name}  (count={pair_count})", font=font, fill=(15, 23, 42))

        sections = [
            ("Misclassified sample", example["path"], (20, 55)),
            (f"Reference: true class ({true_name})", true_ref, (340, 55)),
            (f"Reference: predicted class ({pred_name})", pred_ref, (660, 55)),
        ]
        for title, image_path, (x0, y0) in sections:
            draw.text((x0, y0), title, font=font, fill=(30, 41, 59))
            if image_path and os.path.exists(image_path):
                thumb = _load_thumb(image_path, size=(260, 220))
                panel.paste(thumb, (x0, y0 + 24))
            else:
                draw.rectangle((x0, y0 + 24, x0 + 260, y0 + 244), outline=(148, 163, 184), width=2)
                draw.text((x0 + 76, y0 + 128), "No image", font=font, fill=(100, 116, 139))

        _draw_text_block(
            draw,
            (20, 286),
            [
                f"Example file: {os.path.basename(example['path'])}",
                f"Prediction confidence: {example['confidence']:.4f}",
                "Use this panel to compare morphology/background similarity.",
            ],
            font,
        )
        safe_name = f"{pair_rank:02d}_{true_name}_to_{pred_name}".replace("/", "_").replace(" ", "_")
        panel.save(pair_dir / f"{safe_name}.png")

    print(f"Misclassification analysis saved to {error_dir}")


def load_model(bundle_path: str, backbone_weights: str, device: torch.device):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    backbone = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-5)
    state = torch.load(backbone_weights, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)

    head = _build_head(bundle["head_type"], bundle["embed_dim"], bundle["num_classes"])
    head.load_state_dict(bundle["head_state_dict"])
    head.eval()
    head.to(device)

    return backbone, head, bundle["class_names"]


# ── Inference modes ───────────────────────────────────────────────────

def infer_single_image(backbone, head, class_names, img_path, tf, device):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        features = backbone.forward_features(x)
        logits = _forward_head(head, features)
    elapsed_ms = (time.time() - t0) * 1000

    probs = torch.softmax(logits, dim=1).squeeze()
    top5 = probs.topk(min(5, len(class_names)))

    print(f"\nImage: {img_path}")
    print(f"Inference time: {elapsed_ms:.2f} ms")
    print(f"\n{'Rank':<6} {'Class':<20} {'Confidence':>10}")
    print("-" * 38)
    for rank, (prob, idx) in enumerate(zip(top5.values, top5.indices), 1):
        print(f"{rank:<6} {class_names[idx]:<20} {prob.item():>10.4f}")


def infer_directory(backbone, head, class_names, dir_path, tf, device, batch_size, num_workers, save_csv):
    paths = sorted([
        p for p in Path(dir_path).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not paths:
        print(f"No images found in {dir_path}")
        return

    dataset = ImageListDataset(paths, tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_paths, all_preds, all_confs = [], [], []
    total_time = 0.0

    with torch.no_grad():
        for imgs, img_paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            t0 = time.time()
            features = backbone.forward_features(imgs)
            logits = _forward_head(head, features)
            total_time += time.time() - t0

            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            all_paths.extend(img_paths)
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    n = len(all_paths)
    avg_ms = (total_time / n) * 1000

    print(f"\nProcessed {n} images from {dir_path}")
    print(f"Total inference time: {total_time:.2f}s  |  Avg per image: {avg_ms:.2f} ms\n")

    print(f"{'Image':<60} {'Prediction':<20} {'Confidence':>10}")
    print("-" * 92)
    for path, pred, conf in zip(all_paths, all_preds, all_confs):
        name = os.path.basename(path)
        print(f"{name:<60} {class_names[pred]:<20} {conf:>10.4f}")

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "prediction", "confidence"])
            for path, pred, conf in zip(all_paths, all_preds, all_confs):
                w.writerow([os.path.basename(path), class_names[pred], f"{conf:.4f}"])
        print(f"\nPredictions saved to {save_csv}")


def infer_with_labels(
    backbone,
    head,
    class_names,
    dir_path,
    tf,
    device,
    batch_size,
    num_workers,
    save_csv,
    save_cm,
    save_error_dir,
    max_error_examples,
):
    from sklearn.metrics import classification_report, confusion_matrix

    test_dataset = datasets.ImageFolder(dir_path, transform=tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_class_names = test_dataset.classes

    test_to_train = {}
    for tidx, tname in enumerate(test_class_names):
        base = tname.replace("-compet", "")
        if base in class_names:
            test_to_train[tidx] = class_names.index(base)
        else:
            print(f"  WARNING: test class '{tname}' has no train match, mapping by index")
            test_to_train[tidx] = tidx

    all_preds, all_gt, all_confs = [], [], []
    total_time = 0.0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            t0 = time.time()
            features = backbone.forward_features(imgs)
            logits = _forward_head(head, features)
            total_time += time.time() - t0

            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for pred, conf, label in zip(preds.cpu(), confs.cpu(), labels):
                mapped = test_to_train.get(label.item(), label.item())
                all_gt.append(mapped)
                all_preds.append(pred.item())
                all_confs.append(conf.item())

    all_paths = [s[0] for s in test_dataset.samples]

    n = len(all_gt)
    correct = sum(p == g for p, g in zip(all_preds, all_gt))
    avg_ms = (total_time / n) * 1000

    print(f"\n{'=' * 70}")
    print(f"Test set: {dir_path}  ({n} images)")
    print(f"{'=' * 70}")
    print(f"\nOverall accuracy: {correct / n:.4f} ({correct}/{n})")
    print(f"Avg inference time per image: {avg_ms:.2f} ms\n")

    print("Per-class report:")
    print(classification_report(all_gt, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_gt, all_preds)
    print("Confusion matrix:")
    print(cm)

    cm_path = Path(save_cm) if save_cm else None
    if cm_path is None and save_csv:
        save_csv_path = Path(save_csv)
        cm_path = save_csv_path.with_name(f"{save_csv_path.stem}_confusion_matrix.csv")
    if cm_path is None:
        cm_path = Path.cwd() / "confusion_matrix.csv"

    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for class_name, row in zip(class_names, cm.tolist()):
            writer.writerow([class_name] + row)
    print(f"Confusion matrix saved to {cm_path}")

    misclassified = [
        {
            "path": all_paths[i],
            "ground_truth": class_names[all_gt[i]],
            "prediction": class_names[all_preds[i]],
            "confidence": all_confs[i],
        }
        for i in range(n) if all_preds[i] != all_gt[i]
    ]
    if misclassified:
        print(f"\nMisclassified images ({len(misclassified)}):")
        print(f"{'Image':<55} {'True':<18} {'Predicted':<18} {'Conf':>6}")
        print("-" * 100)
        for item in misclassified:
            print(
                f"{os.path.basename(item['path']):<55} "
                f"{item['ground_truth']:<18} {item['prediction']:<18} {item['confidence']:>6.4f}"
            )

        pair_counts = Counter((item["ground_truth"], item["prediction"]) for item in misclassified)
        print("\nError analysis:")
        print("Most common confusion pairs:")
        for (true_name, pred_name), count in pair_counts.most_common(5):
            avg_conf = sum(
                item["confidence"] for item in misclassified
                if item["ground_truth"] == true_name and item["prediction"] == pred_name
            ) / count
            print(f"  - {true_name} -> {pred_name}: {count} samples, avg confidence {avg_conf:.4f}")

        lowest_conf = sorted(misclassified, key=lambda x: x["confidence"])[: min(5, len(misclassified))]
        print("Lowest-confidence mistakes:")
        for item in lowest_conf:
            print(
                f"  - {os.path.basename(item['path'])}: "
                f"{item['ground_truth']} -> {item['prediction']} ({item['confidence']:.4f})"
            )

        error_dir = _get_error_dir(save_error_dir, save_csv)
        save_misclassification_analysis(misclassified, test_dataset, error_dir, max_error_examples)

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "ground_truth", "prediction", "confidence", "correct"])
            for i in range(n):
                w.writerow([
                    os.path.basename(all_paths[i]),
                    class_names[all_gt[i]],
                    class_names[all_preds[i]],
                    f"{all_confs[i]:.4f}",
                    all_preds[i] == all_gt[i],
                ])
        print(f"\nPredictions saved to {save_csv}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone, head, class_names = load_model(args.bundle, args.backbone_weights, device)
    tf = build_transform(img_size=224)

    input_path = args.input

    if os.path.isfile(input_path):
        infer_single_image(backbone, head, class_names, input_path, tf, device)
    elif os.path.isdir(input_path):
        if args.has_labels:
            infer_with_labels(backbone, head, class_names, input_path, tf, device,
                              args.batch_size, args.num_workers, args.save_csv, args.save_cm,
                              args.save_error_dir, args.max_error_examples)
        else:
            infer_directory(backbone, head, class_names, input_path, tf, device,
                            args.batch_size, args.num_workers, args.save_csv)
    else:
        print(f"ERROR: '{input_path}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
