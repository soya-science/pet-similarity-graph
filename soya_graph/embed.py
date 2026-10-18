"""Load Oxford-IIIT Pets, write thumbnails, and compute CLIP image embeddings (inference only)."""

import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

# The 12 cat breeds in Oxford-IIIT Pets; the other 25 classes are dogs.
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}


class PetImages(Dataset):
    """Wraps the pet images: returns the CLIP input tensor and label, and saves a square thumbnail as a side effect."""

    def __init__(self, base, preprocess, thumb_dir, thumb_size):
        self.base = base
        self.preprocess = preprocess
        self.thumb_dir = thumb_dir
        self.thumb_size = thumb_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        img = img.convert("RGB")
        thumb = ImageOps.fit(img, (self.thumb_size, self.thumb_size))
        thumb.save(self.thumb_dir / f"{i}.jpg", quality=85)
        return self.preprocess(img), label


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pets(data_root, limit, seed):
    """Both official splits joined together, optionally subsampled to `limit` images."""
    splits = [OxfordIIITPet(root=data_root, split=s, download=True) for s in ("trainval", "test")]
    classes = splits[0].classes
    full = ConcatDataset(splits)
    indices = np.arange(len(full))
    if limit and limit < len(full):
        indices = np.sort(np.random.default_rng(seed).choice(len(full), size=limit, replace=False))
    return Subset(full, indices.tolist()), classes


def load_or_embed(
    *,
    model_name,
    pretrained,
    limit,
    seed,
    data_root=Path("data"),
    cache_dir=Path("cache"),
    thumb_dir=Path("web/thumbs"),
    thumb_size=256,
    batch_size=64,
    num_workers=4,
    force=False,
):
    """Return (embeddings [N, D] float32 L2-normalized, meta dict). Reuses the cache when the settings match."""
    emb_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "meta.json"
    key = {"model": model_name, "pretrained": pretrained, "limit": limit, "seed": seed, "thumb_size": thumb_size}

    if not force and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["key"] == key:
            print(f"Using cached embeddings from {emb_path} ({meta['n']} images)")
            return np.load(emb_path), meta

    dataset, classes = load_pets(data_root, limit, seed)

    device = pick_device()
    print(f"Embedding {len(dataset)} images with {model_name} ({pretrained}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.eval().to(device)

    thumb_dir.mkdir(parents=True, exist_ok=True)
    for old in thumb_dir.glob("*.jpg"):
        old.unlink()

    loader = DataLoader(
        PetImages(dataset, preprocess, thumb_dir, thumb_size),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    feats, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, unit="batch"):
            f = model.encode_image(images.to(device))
            feats.append(F.normalize(f.float(), dim=-1).cpu())
            labels.append(batch_labels)
    emb = torch.cat(feats).numpy().astype(np.float32)

    meta = {
        "key": key,
        "n": len(emb),
        "classes": classes,
        "class_species": ["cat" if c in CAT_BREEDS else "dog" for c in classes],
        "labels": torch.cat(labels).tolist(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    meta_path.write_text(json.dumps(meta))
    return emb, meta
