"""
Download NIH ChestX-ray14 dataset to ${DATA_DIR} (default: data/nih_cxr14)
and produce labels.csv in the format expected by CSRDataset.

NIH ChestX-ray14:
  - 112,120 frontal chest X-ray images
  - 14 disease labels (multi-label)
  - Used as a proxy for TBX11K/VinDr-CXR (same domain, same concept types)
  - Binary task: No Finding (0) vs Any Finding (1) — maps to CSR class_label

We download up to MAX_SAMPLES images to keep training time reasonable on A100.
"""

import os
import csv
import sys
from PIL import Image
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
# Paths/sizes are taken from environment variables so the script runs anywhere.
DATA_ROOT   = os.environ.get("DATA_DIR", "data/nih_cxr14")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "20000"))  # 20k images (paper's NIH shard)
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.10
# Remaining 10% → test

# NIH 14 finding labels — used as concept labels
FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

os.makedirs(os.path.join(DATA_ROOT, "images"), exist_ok=True)


def run():
    print(f"Downloading NIH ChestX-ray14 (up to {MAX_SAMPLES} samples)...")
    print(f"Output: {DATA_ROOT}")

    ds = load_dataset(
        "BahaaEldin0/NIH-Chest-Xray-14",
        split="train",
        streaming=True,
    )

    rows = []
    n_saved = 0

    for sample in ds:
        if n_saved >= MAX_SAMPLES:
            break

        # Image
        img: Image.Image = sample["image"]
        image_id = f"nih_{n_saved:06d}"
        img_path = os.path.join(DATA_ROOT, "images", image_id + ".png")
        img.convert("RGB").save(img_path)

        # Concept labels — multi-hot over 14 findings
        label_list = sample["label"]    # e.g. ['Effusion', 'Infiltration']
        if not isinstance(label_list, list):
            label_list = [label_list]

        concept_vec = [1 if f in label_list else 0 for f in FINDINGS]

        # Class label: 0 = No Finding, 1 = Any Finding
        no_finding = (label_list == ["No Finding"] or label_list == [])
        class_label = 0 if no_finding else 1

        # Split assignment
        r = n_saved / MAX_SAMPLES
        if r < TRAIN_FRAC:
            split = "train"
        elif r < TRAIN_FRAC + VAL_FRAC:
            split = "val"
        else:
            split = "test"

        row = {"image_id": image_id, "class_label": class_label, "split": split}
        for i, f in enumerate(FINDINGS):
            row[f"concept_{i}"] = concept_vec[i]
        rows.append(row)

        n_saved += 1
        if n_saved % 500 == 0:
            print(f"  {n_saved}/{MAX_SAMPLES} saved...", flush=True)

    # Write labels.csv
    labels_path = os.path.join(DATA_ROOT, "labels.csv")
    fieldnames = ["image_id", "class_label", "split"] + [f"concept_{i}" for i in range(14)]
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    train_n = sum(1 for r in rows if r["split"] == "train")
    val_n   = sum(1 for r in rows if r["split"] == "val")
    test_n  = sum(1 for r in rows if r["split"] == "test")
    pos_n   = sum(1 for r in rows if r["class_label"] == 1)
    print(f"\nDone. Saved {n_saved} images.")
    print(f"  Train: {train_n}  Val: {val_n}  Test: {test_n}")
    print(f"  Positive (findings): {pos_n} ({100*pos_n/n_saved:.1f}%)")
    print(f"  Labels saved to: {labels_path}")


if __name__ == "__main__":
    run()
