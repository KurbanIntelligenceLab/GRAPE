"""
Download NIH ChestX-ray14 with patient-level train/val/test split.

Differences from download_nih.py (stream-based):
  - Downloads a larger fixed subset (MAX_PATIENTS patients × their images)
  - Extracts patient_id from image metadata (field "img_id" or filename)
  - Splits at patient level → no patient leaks between train/val/test
  - Saves patient_id column in labels.csv for reproducibility

Output (default, override with DATA_DIR):
  data/nih_official/
    images/        JPEG images
    labels.csv     image_id, class_label, patient_id, split, concept_0..13
"""

import os, csv, sys
from collections import defaultdict
from PIL import Image, ImageFile
from datasets import load_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT     = os.environ.get("DATA_DIR", "data/nih_official")
MAX_PATIENTS  = 10000      # enough for patient-level split; ~22-30k images
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# Remaining 15% → test
SEED          = 42

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

os.makedirs(os.path.join(DATA_ROOT, "images"), exist_ok=True)


def extract_patient_id(sample, idx):
    """Try to extract patient ID from sample metadata; fall back to idx-based groups."""
    # HF dataset may expose original filename in several fields
    for key in ("img_id", "image_id", "file_name", "filename", "id"):
        val = sample.get(key, None)
        if val is not None:
            val = str(val)
            # NIH images: "00000001_000.png" — first segment is patient ID
            parts = val.replace(".png", "").replace(".jpg", "").split("_")
            if len(parts) >= 2 and parts[0].isdigit():
                return parts[0]
    # Fallback: group every 3 consecutive images into one "patient"
    return str(idx // 3)


def run():
    import random
    rng = random.Random(SEED)

    print(f"Streaming NIH ChestX-ray14 (patient-level split, up to {MAX_PATIENTS} patients)...")
    ds = load_dataset(
        "BahaaEldin0/NIH-Chest-Xray-14",
        split="train",
        streaming=True,
    )

    # ── Pass 1: collect all samples, group by patient ──────────────────────────
    patient_images = defaultdict(list)   # patient_id → list of sample dicts
    n_streamed = 0
    MAX_STREAM  = MAX_PATIENTS * 5       # upper bound to avoid infinite loop

    for idx, sample in enumerate(ds):
        if len(patient_images) >= MAX_PATIENTS and n_streamed > MAX_PATIENTS * 3:
            break
        if n_streamed >= MAX_STREAM:
            break

        pid = extract_patient_id(sample, idx)
        if len(patient_images) >= MAX_PATIENTS and pid not in patient_images:
            # Don't add more new patients once limit reached
            n_streamed += 1
            continue

        patient_images[pid].append({
            "sample": sample,
            "global_idx": idx,
        })
        n_streamed += 1

        if n_streamed % 2000 == 0:
            print(f"  Streamed {n_streamed} samples, {len(patient_images)} patients so far...",
                  flush=True)

    print(f"  Total patients: {len(patient_images)}, total images: {n_streamed}")

    # ── Patient-level split ────────────────────────────────────────────────────
    patients = sorted(patient_images.keys())
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_pids = set(patients[:n_train])
    val_pids   = set(patients[n_train:n_train + n_val])
    test_pids  = set(patients[n_train + n_val:])

    print(f"  Patient split: train={len(train_pids)}  val={len(val_pids)}  test={len(test_pids)}")

    # ── Pass 2: save images and build CSV ─────────────────────────────────────
    rows     = []
    img_idx  = 0

    for pid, entries in patient_images.items():
        if pid in train_pids:
            split = "train"
        elif pid in val_pids:
            split = "val"
        else:
            split = "test"

        for entry in entries:
            sample     = entry["sample"]
            image_id   = f"nih_{img_idx:06d}"
            jpg_path   = os.path.join(DATA_ROOT, "images", image_id + ".jpg")

            if not os.path.exists(jpg_path):
                try:
                    img: Image.Image = sample["image"]
                    img.convert("RGB").save(jpg_path, format="JPEG", quality=92)
                except Exception as e:
                    print(f"  WARNING: skipping {image_id}: {e}")
                    img_idx += 1
                    continue

            label_list = sample.get("label", [])
            if not isinstance(label_list, list):
                label_list = [label_list]
            concept_vec   = [1 if f in label_list else 0 for f in FINDINGS]
            no_finding    = (label_list in (["No Finding"], []))
            class_label   = 0 if no_finding else 1

            row = {"image_id": image_id, "class_label": class_label,
                   "patient_id": pid, "split": split}
            for i, f in enumerate(FINDINGS):
                row[f"concept_{i}"] = concept_vec[i]
            rows.append(row)
            img_idx += 1

        if img_idx % 2000 == 0:
            print(f"  Saved {img_idx} images...", flush=True)

    # ── Write labels.csv ──────────────────────────────────────────────────────
    labels_path = os.path.join(DATA_ROOT, "labels.csv")
    fieldnames  = (["image_id", "class_label", "patient_id", "split"]
                   + [f"concept_{i}" for i in range(14)])
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_n = sum(1 for r in rows if r["split"] == "train")
    val_n   = sum(1 for r in rows if r["split"] == "val")
    test_n  = sum(1 for r in rows if r["split"] == "test")
    pos_n   = sum(1 for r in rows if r["class_label"] == 1)
    print(f"\nDone. Saved {img_idx} images.")
    print(f"  Train: {train_n}  Val: {val_n}  Test: {test_n}")
    print(f"  Positive (findings): {pos_n} ({100*pos_n/max(img_idx,1):.1f}%)")
    print(f"  Labels → {labels_path}")


if __name__ == "__main__":
    run()
