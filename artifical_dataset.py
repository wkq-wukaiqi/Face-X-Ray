import csv
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


DEFAULT_IMAGE_SIZE = 1024


class ArtifactDataset(Dataset):
    """Resize still images to a square and apply JPEG compression."""

    def __init__(self, root_dir, jpeg_quality=90, image_size=DEFAULT_IMAGE_SIZE):
        if not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 95:
            raise ValueError('jpeg_quality must be an integer from 1 to 95')

        root = Path(root_dir)
        if not root.is_dir():
            raise NotADirectoryError(root)

        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        self.image_paths = sorted(
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise ValueError(f'No supported images found in {root}')

        self.jpeg_quality = jpeg_quality
        self.image_size = image_size
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.to_tensor(load_portrait(self.image_paths[index], self.jpeg_quality, self.image_size))


def load_portrait(path, jpeg_quality=90, image_size=DEFAULT_IMAGE_SIZE):
    """Apply the same resize and optional JPEG encoding to either class."""
    with Image.open(path) as image:
        portrait = ImageOps.exif_transpose(image).convert("RGB")
        size = (image_size, image_size)
        if portrait.size != size:
            portrait = portrait.resize(size, Image.Resampling.BILINEAR)
    if jpeg_quality is None:
        return portrait
    with BytesIO() as buffer:
        portrait.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class LabeledArtifactDataset(Dataset):
    """Read whole-portrait samples with labels 0 (real) and 1 (synthetic)."""

    def __init__(self, manifest, split, jpeg_quality=90, image_size=DEFAULT_IMAGE_SIZE):
        manifest = Path(manifest).resolve()
        self.jpeg_quality = jpeg_quality
        self.image_size = image_size
        self.to_tensor = ToTensor()
        self.samples = []
        group_splits = {}
        paths = set()
        splits = {"train", "val", "test"}
        if split not in splits:
            raise ValueError(f"Unknown split: {split}")
        if jpeg_quality is not None and not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")

        with manifest.open(newline="", encoding="utf-8-sig") as csv_file:
            rows = csv.DictReader(csv_file)
            required = {"path", "label", "split", "source_group"}
            if not required.issubset(rows.fieldnames or []):
                raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
            for row in rows:
                path, label, row_split, group = (
                    (row[key] or "").strip() for key in ("path", "label", "split", "source_group")
                )
                if not path or not group or row_split not in splits or label not in {"0", "1"}:
                    raise ValueError(f"Invalid manifest row {rows.line_num}")
                path = (manifest.parent / path).resolve()
                if path in paths:
                    raise ValueError(f"Duplicate manifest path: {path}")
                paths.add(path)
                if group_splits.setdefault(group, row_split) != row_split:
                    raise ValueError(f"Source group occurs in multiple splits: {group}")
                if row_split == split:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    self.samples.append((path, int(label)))

        if {label for _, label in self.samples} != {0, 1}:
            raise ValueError(f"The {split} split must contain real and synthetic portraits")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.to_tensor(load_portrait(path, self.jpeg_quality, self.image_size)), label


def get_artifact_loader(root_dir, jpeg_quality=90, batch_size=1, image_size=DEFAULT_IMAGE_SIZE):
    """Return batches of square RGB tensors."""
    dataset = ArtifactDataset(root_dir, jpeg_quality=jpeg_quality, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size)
