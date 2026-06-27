"""
Download NIH ChestX-ray14 images that have bounding box annotations.
Saves images with patient-ID-based filenames so they can be matched
to BBox_List_2017.csv entries.

Output directory: ${DATA_DIR} (default: data/nih_bbox)
  images/         ← {PatientID:08d}_{localidx:03d}.png
  labels.csv      ← CSRDataset-compatible
  bboxes.csv      ← concept_idx, x1, y1, x2, y2 in 224x224 space

NIH BBox format: x, y, w, h in 1024x1024 image space
We scale to 224x224 (the model's input size) and convert to x1,y1,x2,y2.

Finding label → concept_idx mapping:
  Atelectasis=0, Cardiomegaly=1, Consolidation=2, Edema=3, Effusion=4,
  Emphysema=5, Fibrosis=6, Hernia=7, Infiltration=8, Mass=9, Nodule=10,
  Pleural_Thickening=11, Pneumonia=12, Pneumothorax=13
  (Infiltrate in bbox file maps to Infiltration → idx 8)
"""

import os, csv, sys
from PIL import Image
from datasets import load_dataset

# Paths are taken from environment variables so the script runs anywhere.
BBOX_CSV    = os.environ.get("BBOX_CSV", "data/nih_cxr14/BBox_List_2017.csv")
DATA_ROOT   = os.environ.get("DATA_DIR", "data/nih_bbox")
ORIG_SIZE   = 1024
TARGET_SIZE = 224
SCALE       = TARGET_SIZE / ORIG_SIZE   # 0.21875

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]
# BBox file uses "Infiltrate" but our concept is "Infiltration"
LABEL_MAP = {f: f for f in FINDINGS}
LABEL_MAP["Infiltrate"] = "Infiltration"

CLASS_NAMES = ["no_finding", "finding"]

os.makedirs(os.path.join(DATA_ROOT, "images"), exist_ok=True)

# ── Step 1: Parse bbox file ────────────────────────────────────────────────
print("Parsing BBox_List_2017.csv...")
bbox_by_patient = {}   # patient_id (int) → list of (finding, x, y, w, h)
with open(BBOX_CSV) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        img_index = row[0]          # e.g. 00013118_008.png
        finding   = row[1].strip()
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        patient_id = int(img_index.split("_")[0])
        bbox_by_patient.setdefault(patient_id, []).append((finding, x, y, w, h))

target_patients = set(bbox_by_patient.keys())
print(f"  {len(target_patients)} patients with bbox annotations")

# ── Step 2: Stream HF dataset, save matching patients ─────────────────────
print("Streaming HF dataset for matching patients...")
ds = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", split="train", streaming=True)

# Track how many images we've saved per patient (for filename uniqueness)
patient_count = {}
saved_images  = []   # list of (image_id, patient_id, label_list)

for sample in ds:
    pid = sample["Patient ID"]
    if pid not in target_patients:
        continue

    count = patient_count.get(pid, 0)
    image_id = f"{pid:08d}_{count:03d}"
    patient_count[pid] = count + 1

    img_path = os.path.join(DATA_ROOT, "images", image_id + ".png")
    sample["image"].convert("RGB").save(img_path)

    label_list = sample["label"]
    if not isinstance(label_list, list):
        label_list = [label_list]

    saved_images.append((image_id, pid, label_list))

    if len(saved_images) % 100 == 0:
        print(f"  {len(saved_images)} images saved "
              f"({len(patient_count)}/{len(target_patients)} patients)...", flush=True)

    # Stop once we have at least one image per patient
    if len(patient_count) >= len(target_patients) and all(
        patient_count.get(p, 0) >= 1 for p in target_patients
    ):
        print("  All target patients covered — stopping stream.")
        break

print(f"Total images saved: {len(saved_images)}")

# ── Step 3: Write labels.csv ───────────────────────────────────────────────
print("Writing labels.csv...")
label_rows = []
for i, (image_id, pid, label_list) in enumerate(saved_images):
    no_finding = (label_list == ["No Finding"] or label_list == [])
    class_label = 0 if no_finding else 1

    concept_vec = [0] * 14
    for lbl in label_list:
        mapped = LABEL_MAP.get(lbl, lbl)
        if mapped in FINDINGS:
            concept_vec[FINDINGS.index(mapped)] = 1

    r = i / len(saved_images)
    split = "train" if r < 0.7 else ("val" if r < 0.85 else "test")

    row = {"image_id": image_id, "class_label": class_label, "split": split}
    for j in range(14):
        row[f"concept_{j}"] = concept_vec[j]
    label_rows.append((image_id, pid, row))

labels_path = os.path.join(DATA_ROOT, "labels.csv")
fieldnames = ["image_id", "class_label", "split"] + [f"concept_{i}" for i in range(14)]
with open(labels_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([r for _, _, r in label_rows])

# ── Step 4: Write bboxes.csv ───────────────────────────────────────────────
# For each patient, assign their bbox annotations to the FIRST saved image.
# (Best approximation without original filename → image number mapping.)
print("Writing bboxes.csv...")
patient_to_imageid = {}
for image_id, pid, _ in label_rows:
    if pid not in patient_to_imageid:
        patient_to_imageid[pid] = image_id

bbox_rows = []
skipped = 0
for pid, annots in bbox_by_patient.items():
    if pid not in patient_to_imageid:
        skipped += 1
        continue
    image_id = patient_to_imageid[pid]
    for (finding, x, y, w, h) in annots:
        mapped = LABEL_MAP.get(finding, finding)
        if mapped not in FINDINGS:
            continue
        concept_idx = FINDINGS.index(mapped)
        # Scale from 1024×1024 → 224×224 and convert (x,y,w,h) → (x1,y1,x2,y2)
        x1 = int(x * SCALE)
        y1 = int(y * SCALE)
        x2 = int((x + w) * SCALE)
        y2 = int((y + h) * SCALE)
        bbox_rows.append({
            "image_id": image_id,
            "concept_idx": concept_idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

bboxes_path = os.path.join(DATA_ROOT, "bboxes.csv")
with open(bboxes_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "concept_idx", "x1", "y1", "x2", "y2"])
    writer.writeheader()
    writer.writerows(bbox_rows)

# ── Summary ────────────────────────────────────────────────────────────────
test_n  = sum(1 for _, _, r in label_rows if r["split"] == "test")
train_n = sum(1 for _, _, r in label_rows if r["split"] == "train")
val_n   = sum(1 for _, _, r in label_rows if r["split"] == "val")
print(f"\nDone.")
print(f"  Images:  {len(saved_images)} (train={train_n}, val={val_n}, test={test_n})")
print(f"  Labels:  {labels_path}")
print(f"  BBoxes:  {len(bbox_rows)} annotations → {bboxes_path}")
print(f"  Skipped: {skipped} patients (not found in stream)")
