"""
Dataset loaders for GRAPE training and evaluation.

Datasets used in the paper:
  - TBX11K: chest X-ray tuberculosis detection, 3 concepts (has bbox annotations)
  - NIH ChestX-ray14: 14 findings as concepts, binary task (has bbox annotations)

NOTE: 'isic' below is UNUSED scaffolding from an earlier project scope, NOT
part of the paper, untested, and lacking a matching config. Retained only as
a starting point for extension and should not be relied upon.

Each dataset returns:
  image:          (3, H, W) normalized tensor
  concept_labels: (K,) binary concept presence labels
  class_label:    int — target class index
  bbox:           dict mapping concept_idx → (x1, y1, x2, y2) [TBX11K only]
  image_id:       str — unique image identifier

Directory layout expected:
  TBX11K/
    images/           ← .jpg/.png files
    labels.csv        ← columns: image_id, class, concept_0, concept_1, ...
    bboxes.csv        ← columns: image_id, concept_idx, x1, y1, x2, y2

  ISIC/
    images/
    labels.csv
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ──────────────────────────────────────────────────────────────────────────────
# Concept definitions per dataset
# ──────────────────────────────────────────────────────────────────────────────

CONCEPTS = {
    # TBX11K: 3 TB finding types from COCO annotations
    "tbx11k": [
        "active_tuberculosis",
        "obsolete_pulmonary_tb",
        "pulmonary_tuberculosis",
    ],
    "isic": [
        "pigment_network", "negative_network", "streaks", "milia_like_cyst",
        "globules", "structureless", "regression_structures", "dots",
    ],
    # NIH ChestX-ray14: 14 official finding labels used as concepts
    "nih": [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax",
    ],
}

CLASS_NAMES = {
    "tbx11k": ["healthy", "sick_but_non_tb", "active_tb"],
    "nih": ["no_finding", "finding"],
    # --- UNUSED (not in paper); kept as extension scaffolding only ---
    "isic": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Image transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet-normalized transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Base dataset
# ──────────────────────────────────────────────────────────────────────────────

class CSRDataset(Dataset):
    """
    Base dataset class for CSR++ experiments.

    Reads a CSV with columns:
      image_id, class_label, concept_0, concept_1, ..., concept_K-1

    Optionally reads a bounding box CSV with columns:
      image_id, concept_idx, x1, y1, x2, y2
    """

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str = "train",
        image_size: int = 224,
        labels_file: str = "labels.csv",
        bboxes_file: str = "bboxes.csv",
    ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.concepts = CONCEPTS[dataset_name]
        self.class_names = CLASS_NAMES[dataset_name]
        self.num_concepts = len(self.concepts)
        self.num_classes = len(self.class_names)
        # bbox_eval uses val-style transforms (no augmentation)
        transform_split = "val" if split == "bbox_eval" else split
        self.transform = get_transforms(transform_split, image_size)

        self.samples = []        # list of dicts: image_id, class_label, concept_labels
        self.bboxes = {}         # image_id → dict(concept_idx → (x1,y1,x2,y2))

        self._load_labels(os.path.join(root_dir, labels_file))

        bbox_path = os.path.join(root_dir, bboxes_file)
        if os.path.exists(bbox_path):
            self._load_bboxes(bbox_path)

    def _load_labels(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Labels file not found: {path}")

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "split" in row and row["split"] != self.split:
                    continue
                concept_labels = torch.tensor(
                    [float(row.get(f"concept_{i}", 0)) for i in range(self.num_concepts)],
                    dtype=torch.float32,
                )
                self.samples.append({
                    "image_id": row["image_id"],
                    "class_label": int(row["class_label"]),
                    "concept_labels": concept_labels,
                })

    def _load_bboxes(self, path: str):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = row["image_id"]
                if iid not in self.bboxes:
                    self.bboxes[iid] = {}
                self.bboxes[iid][int(row["concept_idx"])] = (
                    int(row["x1"]), int(row["y1"]),
                    int(row["x2"]), int(row["y2"]),
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_id = sample["image_id"]

        # Load image — try common extensions
        img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(self.root_dir, "images", image_id + ext)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                break

        if img is None:
            raise FileNotFoundError(f"Image not found for id: {image_id}")

        image = self.transform(img)

        return {
            "image": image,
            "class_label": torch.tensor(sample["class_label"], dtype=torch.long),
            "concept_labels": sample["concept_labels"],
            "bbox": self.bboxes.get(image_id, {}),
            "image_id": image_id,
        }

    def get_all_concept_labels(self) -> torch.Tensor:
        """Return full concept label matrix (N, K) — used to build the co-occurrence graph."""
        return torch.stack([s["concept_labels"] for s in self.samples])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = "train",
    image_size: int = 224,
) -> CSRDataset:
    """
    Factory function — returns the correct dataset for the given name.

    Args:
        dataset_name: one of 'tbx11k', 'nih', 'isic'
        root_dir:     path to the dataset root directory
        split:        'train', 'val', or 'test'
        image_size:   resize target

    Returns:
        CSRDataset instance
    """
    assert dataset_name in CONCEPTS, f"Unknown dataset: {dataset_name}. Choose from {list(CONCEPTS)}"
    return CSRDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
    )


def get_dataloader(
    dataset: CSRDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """
    Wrap a CSRDataset in a DataLoader with sensible defaults.

    shuffle defaults to True for train, False otherwise.
    """
    if shuffle is None:
        shuffle = (dataset.split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: stack tensors, keep bbox dicts as a list,
    and keep image_id as a list of strings.
    """
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "class_label": torch.stack([b["class_label"] for b in batch]),
        "concept_labels": torch.stack([b["concept_labels"] for b in batch]),
        "bbox": [b["bbox"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset for unit testing (no files needed)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """
    Generates random tensors in the correct format for unit testing
    the training pipeline without any real data files.
    """

    def __init__(
        self,
        dataset_name: str = "tbx11k",
        num_samples: int = 64,
        image_size: int = 224,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.concepts = CONCEPTS[dataset_name]
        self.class_names = CLASS_NAMES[dataset_name]
        self.num_concepts = len(self.concepts)
        self.num_classes = len(self.class_names)
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split

        # Pre-generate fixed random labels for reproducibility
        torch.manual_seed(0)
        self._images = torch.randn(num_samples, 3, image_size, image_size)
        self._concept_labels = torch.randint(0, 2, (num_samples, self.num_concepts)).float()
        self._class_labels = torch.randint(0, self.num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self._images[idx],
            "class_label": self._class_labels[idx],
            "concept_labels": self._concept_labels[idx],
            "bbox": {},
            "image_id": f"synthetic_{idx:05d}",
        }

    def get_all_concept_labels(self) -> torch.Tensor:
        return self._concept_labels
