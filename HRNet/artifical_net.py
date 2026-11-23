import math

import torch
from torch import nn
from torch.nn import functional


def local_residual(image: torch.Tensor):
    downsampled = functional.avg_pool2d(image, kernel_size=2, stride=2)
    smooth = functional.interpolate(
        downsampled, size=image.shape[-2:], mode="bilinear", align_corners=False
    )
    return image - smooth


def patch_layers():
    return [
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
    ]


class ArtifactNet(nn.Module):
    """Return one logit per portrait and a spatial grid of patch logits."""

    def __init__(self, input_mode="residual", top_fraction=1.0):
        super().__init__()
        if input_mode not in {"rgb", "residual"}:
            raise ValueError("input_mode must be rgb or residual")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        self.input_mode = input_mode
        self.top_fraction = top_fraction
        self.patch_net = nn.Sequential(
            *patch_layers(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, image):
        if self.input_mode == "residual":
            image = local_residual(image)
        patch_logits = self.patch_net(image)
        patches = patch_logits.flatten(1)
        if self.top_fraction < 1:
            count = math.ceil(self.top_fraction * patches.size(1))
            patches = patches.topk(count, dim=1, sorted=False).values
        return patches.mean(dim=1), patch_logits


class MultiscaleArtifactNet(nn.Module):
    """Mean-pool residual features at each resolution before classification."""

    def __init__(self, resolutions=(256, 1024)):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.branches = nn.ModuleList(nn.Sequential(*patch_layers()) for _ in self.resolutions)
        self.classifier = nn.Linear(64 * len(self.resolutions), 1)

    def forward(self, images):
        features = []
        for resolution, branch in zip(self.resolutions, self.branches):
            size = (resolution, resolution)
            scaled = images
            if images.shape[-2:] != size:
                scaled = functional.interpolate(images, size=size, mode="bilinear",
                                                align_corners=False, antialias=True)
            features.append(branch(local_residual(scaled)).mean(dim=(-2, -1)))
        logits = self.classifier(torch.cat(features, dim=1)).squeeze(1)
        return logits, None
