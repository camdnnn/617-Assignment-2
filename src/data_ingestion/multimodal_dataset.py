#!/usr/bin/env python3
"""
Origin:
-------
The initial data ingestion was prototyped in a Jupyter notebook using:
- torchvision.datasets.ImageFolder
- Resize(224, 224)
- RandomHorizontalFlip (train only)
- ToTensor()

That notebook version handled image-only classification.

Now:

Multimodal ingestion for the CVPR_2024 garbage dataset.

Returns batches with:
- pixel_values: FloatTensor [B, 3, 224, 224]
- input_ids: LongTensor [B, L]
- attention_mask: LongTensor [B, L]
- label: LongTensor [B]

Text is derived from the filename stem (with cleaning).
Classes are fixed to dataset folder names: Black, Blue, Green, TTR
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from transformers import AutoTokenizer


CLASS_FOLDERS = ["Black", "Blue", "Green", "TTR"]
CLASS_TOKENS = {c.lower() for c in CLASS_FOLDERS}


def filename_to_sentence(stem: str) -> str:
    """
    Convert a filename stem into a short sentence.

    Handles:
      - underscores and spaces
      - trailing numeric ids (e.g., _692)
      - leading class tokens inside filename (e.g., 'Blue_Hair_Brush_692' in TTR)
    """
    s = stem.replace("_", " ")

    # Remove trailing numeric id (e.g. "hair brush 692")
    s = re.sub(r"\s*\d+$", "", s)

    s = s.lower().strip()

    # Remove leading class token to avoid leakage via text
    parts = s.split()
    if parts and parts[0] in CLASS_TOKENS:
        parts = parts[1:]
    s = " ".join(parts)

    # Collapse extra whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


class GarbageMultimodalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        *,
        is_train: bool,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 32,
        local_files_only: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.max_length = max_length

        self.classes = list(CLASS_FOLDERS)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # Gather samples
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((p, self.class_to_idx[cls]))

        # Transforms (same spirit as your notebook)
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=local_files_only
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Text from filename
        sentence = filename_to_sentence(img_path.stem)

        enc = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def make_loaders(
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 32,
    local_files_only: bool = False,
):
    train_ds = GarbageMultimodalDataset(
        train_path,
        is_train=True,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        local_files_only=local_files_only,
    )
    val_ds = GarbageMultimodalDataset(
        val_path,
        is_train=False,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        local_files_only=local_files_only,
    )
    test_ds = GarbageMultimodalDataset(
        test_path,
        is_train=False,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        local_files_only=local_files_only,
    )

    print("Class → Index Mapping:")
    print(train_ds.class_to_idx)
    print("Classes:")
    print(train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx