import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from artifical_dataset import DEFAULT_IMAGE_SIZE, LabeledArtifactDataset
from HRNet.artifical_net import ArtifactNet


def train_artifact_epoch(model, loader, optimizer, device):
    """Train on image labels and return the mean loss."""
    model.train()
    loss_sum = 0.0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.numel()
        count += labels.numel()
    if not count:
        raise ValueError("Training loader is empty")
    return loss_sum / count


@torch.inference_mode()
def collect_artifact_scores(model, loader, device):
    """Collect raw image logits and binary labels on CPU."""
    model.eval()
    scores, targets = [], []
    for images, labels in loader:
        logits, _ = model(images.to(device))
        scores.append(logits.cpu())
        targets.append(labels.cpu())
    if not scores:
        raise ValueError("Evaluation loader is empty")
    return torch.cat(scores), torch.cat(targets)


def select_artifact_threshold(scores, labels, target_fpr=0.05):
    """Fit a threshold on real validation scores using score > threshold."""
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    real_scores = scores[labels == 0]
    if not real_scores.numel():
        raise ValueError("Threshold selection requires real validation portraits")
    allowed = math.floor(target_fpr * real_scores.numel())
    # Strict comparison keeps tied scores from exceeding the false-positive budget.
    rank = real_scores.numel() - allowed
    return real_scores.kthvalue(rank).values.item()


def evaluate_artifact(model, loader, device, threshold):
    """Measure synthetic recall and real FPR at a previously selected threshold."""
    scores, labels = collect_artifact_scores(model, loader, device)
    return artifact_metrics(scores, labels, threshold)


def artifact_metrics(scores, labels, threshold):
    flagged = scores > threshold
    real, synthetic = labels == 0, labels == 1
    real_count, synthetic_count = real.sum().item(), synthetic.sum().item()
    if not real_count or not synthetic_count:
        raise ValueError("Evaluation requires real and synthetic portraits")
    false_positives = (flagged & real).sum().item()
    true_positives = (flagged & synthetic).sum().item()
    real_scores = scores[real].sort().values
    synthetic_scores = scores[synthetic].contiguous()
    lower = torch.searchsorted(real_scores, synthetic_scores)
    upper = torch.searchsorted(real_scores, synthetic_scores, right=True)
    return {
        "roc_auc": (lower + upper).to(torch.float64).mean().item() / (2 * real_count),
        "synthetic_recall": true_positives / synthetic_count,
        "synthetic_detected": true_positives,
        "synthetic_count": synthetic_count,
        "real_fpr": false_positives / real_count,
        "real_flagged": false_positives,
        "real_count": real_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and test the portrait artifact detector.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifact_detector.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-mode", choices=("residual", "rgb"), default="residual")
    parser.add_argument("--top-fraction", type=float, default=1.0, help="Use 1 for mean pooling (default)")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.image_size < 2:
        parser.error("--image-size must be at least 2")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        split: DataLoader(
            LabeledArtifactDataset(
                args.manifest, split, jpeg_quality=args.jpeg_quality, image_size=args.image_size
            ),
            batch_size=args.batch_size,
            shuffle=split == "train",
        )
        for split in ("train", "val", "test")
    }
    model_options = {"input_mode": args.input_mode, "top_fraction": args.top_fraction}
    model = ArtifactNet(**model_options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        loss = train_artifact_epoch(model, loaders["train"], optimizer, device)
        print(f"Epoch {epoch}/{args.epochs}  loss {loss:.4f}")

    scores, labels = collect_artifact_scores(model, loaders["val"], device)
    threshold = select_artifact_threshold(scores, labels, args.target_fpr)
    metrics = evaluate_artifact(model, loaders["test"], device, threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_options": model_options,
            "threshold": threshold,
            "jpeg_quality": args.jpeg_quality,
            "image_size": args.image_size,
        },
        args.output,
    )
    print(f"Validation logit threshold {threshold:.6f}")
    print(
        f"Test synthetic recall {metrics['synthetic_recall']:.1%} "
        f"({metrics['synthetic_detected']}/{metrics['synthetic_count']})"
    )
    print(
        f"Test real FPR {metrics['real_fpr']:.1%} "
        f"({metrics['real_flagged']}/{metrics['real_count']})"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
