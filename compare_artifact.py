"""Compare portrait detectors on one fixed manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode, Normalize, PILToTensor, Resize

from artifical_dataset import LabeledArtifactDataset, load_portrait
from HRNet.artifical_net import MultiscaleArtifactNet
from train_artifact import (
    artifact_metrics,
    collect_artifact_scores,
    select_artifact_threshold,
    train_artifact_epoch,
)

SPLITS = ("train", "val", "test")
RESIDUAL_VARIANTS = {"residual-256": (256,), "residual-1024": (1024,),
                     "residual-multiscale": (256, 1024)}


def make_loader(manifest, split, jpeg_quality, batch_size):
    """Decode each image once and keep its compact uint8 representation."""
    to_pixels = PILToTensor()
    dataset = LabeledArtifactDataset(manifest, split, jpeg_quality=jpeg_quality)
    pixels = torch.empty((len(dataset), 3, 1024, 1024), dtype=torch.uint8)
    for index, (path, _) in enumerate(dataset.samples):
        pixels[index] = to_pixels(load_portrait(path, jpeg_quality, 1024))
    labels = torch.tensor([label for _, label in dataset.samples])
    return DataLoader(TensorDataset(pixels, labels), batch_size=batch_size)


def residual_batches(loader, device):
    for pixels, labels in loader:
        yield pixels.to(device).float().div_(255), labels


def fit_logistic(features, labels, regularization=0.001):
    head = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = head(features).squeeze(1)
        loss = functional.binary_cross_entropy_with_logits(logits, labels.float())
        loss = loss + regularization * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return head


@torch.no_grad()
def extract_clip_features(loaders, weights_path, device):
    if device.type != "cuda":
        raise RuntimeError("The OpenAI TorchScript CLIP checkpoint requires CUDA in this runner")
    encoder = torch.jit.load(str(weights_path), map_location=device).eval()
    resize = Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)
    normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    features = {}
    for split, loader in loaders.items():
        batches = []
        for pixels, _ in loader:
            images = pixels.to(device).float().div_(255)
            inputs = normalize(resize(images).clamp_(0, 1))
            embeddings = encoder.encode_image(inputs)
            batches.append(functional.normalize(embeddings.float(), dim=1).cpu())
        features[split] = torch.cat(batches)
    return features


def save_result(output, name, scores, labels, metadata, **state):
    threshold = select_artifact_threshold(scores["val"], labels["val"])
    metrics = {
        split: artifact_metrics(scores[split], labels[split], threshold)
        for split in ("val", "test")
    }
    torch.save(
        {"scores": scores, "labels": labels, "threshold": threshold, "metrics": metrics,
         **metadata, **state},
        output / f"{name}.pth",
    )
    print(f"{name}: {json.dumps(metrics)}", flush=True)


def run_residual(loaders, args, metadata, device):
    labels = {split: loader.dataset.tensors[1] for split, loader in loaders.items()}
    for name, sizes in RESIDUAL_VARIANTS.items():
        if args.variant and name != args.variant:
            continue
        torch.manual_seed(args.seed)
        model = MultiscaleArtifactNet(sizes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train = DataLoader(loaders["train"].dataset, batch_size=args.batch_size, shuffle=True,
                           generator=torch.Generator().manual_seed(args.seed))
        for epoch in range(1, args.epochs + 1):
            loss = train_artifact_epoch(model, residual_batches(train, device), optimizer, device)
            print(f"{name} epoch {epoch}/{args.epochs} loss {loss:.4f}", flush=True)
        scores = {
            split: collect_artifact_scores(model, residual_batches(loader, device), device)[0]
            for split, loader in loaders.items()
        }
        save_result(args.output, name, scores, labels, metadata,
                    resolutions=sizes, state_dict=model.cpu().state_dict())


def run_clip(loaders, args, metadata, device):
    labels = {split: loader.dataset.tensors[1] for split, loader in loaders.items()}
    features = extract_clip_features(loaders, args.clip_weights, device)
    model_name = "OpenAI CLIP ViT-B/32"
    torch.save({"features": features, "labels": labels, "model": model_name, **metadata},
               args.output / "clip-embeddings.pth")
    head = fit_logistic(features["train"], labels["train"])
    scores = {split: head(x).squeeze(1).detach() for split, x in features.items()}
    save_result(args.output, "clip", scores, labels, metadata,
                clip_model=model_name, state_dict=head.state_dict())


def run_fusion(args, metadata):
    def read_result(name):
        result = torch.load(args.output / f"{name}.pth", map_location="cpu", weights_only=True)
        if any(result[key] != value for key, value in metadata.items()):
            raise ValueError(f"{name} was evaluated with different data or preprocessing")
        return result

    candidates = [(name, read_result(name)) for name in RESIDUAL_VARIANTS]
    name, residual = max(candidates, key=lambda item: (
        item[1]["metrics"]["val"]["synthetic_recall"], item[1]["metrics"]["val"]["roc_auc"]
    ))
    clip = read_result("clip")
    labels = residual["labels"]
    inputs = {}
    for split in SPLITS:
        if not torch.equal(labels[split], clip["labels"][split]):
            raise ValueError(f"CLIP and residual {split} labels differ")
        inputs[split] = torch.stack([residual["scores"][split], clip["scores"][split]], dim=1)
    center = inputs["val"].mean(0)
    scale = inputs["val"].std(0, unbiased=False).clamp_min(1e-8)
    normalized = {split: (x - center) / scale for split, x in inputs.items()}
    head = fit_logistic(normalized["val"], labels["val"])
    scores = {split: head(x).squeeze(1).detach() for split, x in normalized.items()}
    print(f"Fusion uses {name} and clip. Validation is the fusion fit set.", flush=True)
    save_result(args.output, "fusion", scores, labels, metadata,
                residual_model=name, center=center, scale=scale, state_dict=head.state_dict())


@torch.no_grad()
def run_external(args, device):
    loader = make_loader(args.manifest, "test", args.jpeg_quality or None, args.batch_size)
    labels = loader.dataset.tensors[1]

    def load_result(name):
        return torch.load(args.output / f"{name}.pth", map_location="cpu", weights_only=True)

    fusion = load_result("fusion")
    residual = load_result(fusion["residual_model"])
    clip = load_result("clip")
    model = MultiscaleArtifactNet(residual["resolutions"]).to(device)
    model.load_state_dict(residual["state_dict"])
    residual_scores, _ = collect_artifact_scores(model, residual_batches(loader, device), device)
    embeddings = extract_clip_features({"test": loader}, args.clip_weights, device)["test"]
    clip_logits = functional.linear(embeddings, clip["state_dict"]["weight"],
                                    clip["state_dict"]["bias"]).squeeze(1)
    pairs = torch.stack((residual_scores, clip_logits), dim=1)
    normalized = (pairs - fusion["center"]) / fusion["scale"]
    fusion_logits = functional.linear(normalized, fusion["state_dict"]["weight"],
                                      fusion["state_dict"]["bias"]).squeeze(1)
    for name, scores, result in ((fusion["residual_model"], residual_scores, residual),
                                 ("clip", clip_logits, clip), ("fusion", fusion_logits, fusion)):
        metrics = artifact_metrics(scores, labels, result["threshold"])
        print(f"{name}: {json.dumps(metrics)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("residual", "clip", "fusion", "external"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=90, help="0 disables JPEG encoding")
    parser.add_argument("--clip-weights", type=Path, help="Local OpenAI ViT-B-32.pt TorchScript weights")
    parser.add_argument("--variant", choices=tuple(RESIDUAL_VARIANTS))
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or not 0 <= args.jpeg_quality <= 95:
        parser.error("Use positive epochs and batch size, and JPEG quality from 0 to 95")
    if args.mode in {"clip", "external"} and (args.clip_weights is None or not args.clip_weights.is_file()):
        parser.error("This mode requires --clip-weights pointing to the local checkpoint")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    metadata = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "jpeg_quality": args.jpeg_quality}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "fusion":
        run_fusion(args, metadata)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "external":
        run_external(args, device)
        return
    loaders = {
        split: make_loader(args.manifest, split, args.jpeg_quality or None, args.batch_size)
        for split in SPLITS
    }
    if args.mode == "residual":
        run_residual(loaders, args, metadata, device)
    else:
        run_clip(loaders, args, metadata, device)


if __name__ == "__main__":
    main()
