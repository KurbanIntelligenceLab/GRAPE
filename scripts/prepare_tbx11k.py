"""
Prepare TBX11K dataset for CSR++ training.

Source: ${TBX11K_SRC} (default: data/tbx11k_raw/TBX11K)
Output: ${DATA_DIR}     (default: data/tbx11k)
  images/       ← symlinks or copies (or we use original paths)
  labels.csv    ← image_id, class_label, split, concept_0..concept_13
  bboxes.csv    ← image_id, concept_idx, x1, y1, x2, y2 (224×224 space)

Class mapping (3-class):
  0 = healthy  (imgs/health/)
  1 = sick_but_non_tb  (imgs/sick/)
  2 = active_tb  (imgs/tb/)
  test/ images are unlabeled — we skip them

Concept mapping: TB findings from XML annotations
  The 3 annotation categories map to a single binary concept "active_tb" (concept_0).
  We use 3 concepts matching CONCEPTS["tbx11k"] subset:
    concept_0: active_tb   (category: ActiveTuberculosis or PulmonaryTuberculosis)
    concept_1: obsolete_tb (category: ObsoletePulmonaryTuberculosis)
    concept_2: any_tb      (union of the above)

  Actually, TBX11K has TB type annotations so we'll use them as separate concepts.

Bbox notes:
  - JSON bboxes are in original image dimensions (variable, not 224)
  - XML bboxes are also in original image dims
  We use the JSON COCO annotations since they include image dimensions.
  Scale: x1 = int(x / img_w * 224), etc.
"""

import os
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths are taken from environment variables so the script runs anywhere.
SRC   = os.environ.get("TBX11K_SRC", "data/tbx11k_raw/TBX11K")
DEST  = os.environ.get("DATA_DIR", "data/tbx11k")
TARGET_SIZE = 224

# TBX11K concepts (TB finding types as binary concepts)
# We keep it simple: concept_0=ActiveTB, concept_1=ObsoletePTB, concept_2=PulmonaryTB
CATEGORY_TO_CONCEPT = {
    "ActiveTuberculosis":          0,
    "ObsoletePulmonaryTuberculosis": 1,
    "PulmonaryTuberculosis":       2,
}

os.makedirs(os.path.join(DEST, "images"), exist_ok=True)

# ── Step 1: Load JSON annotations ────────────────────────────────────────────
print("Loading JSON annotations...")

def load_coco_split(json_path):
    with open(json_path) as f:
        d = json.load(f)
    img_by_id = {img["id"]: img for img in d["images"]}
    # Map image_id → list of annotations
    ann_by_imgid = {}
    for ann in d.get("annotations", []):
        ann_by_imgid.setdefault(ann["image_id"], []).append(ann)
    return img_by_id, ann_by_imgid

train_imgs, train_anns = load_coco_split(os.path.join(SRC, "annotations/json/TBX11K_train.json"))
val_imgs,   val_anns   = load_coco_split(os.path.join(SRC, "annotations/json/TBX11K_val.json"))

print(f"  Train: {len(train_imgs)} images, {sum(len(v) for v in train_anns.values())} annotations")
print(f"  Val:   {len(val_imgs)} images, {sum(len(v) for v in val_anns.values())} annotations")

# Category ID → concept index
# From JSON: id=1→ActiveTuberculosis, id=2→ObsoletePulmonaryTuberculosis, id=3→PulmonaryTuberculosis
CAT_ID_TO_CONCEPT = {1: 0, 2: 1, 3: 2}

def get_class_from_path(file_name):
    """Map file_name prefix to class label."""
    prefix = file_name.split("/")[0]
    return {"health": 0, "sick": 1, "tb": 2}.get(prefix, -1)

def process_split(img_by_id, ann_by_imgid, split_name):
    """Returns list of label rows and list of bbox rows."""
    label_rows = []
    bbox_rows  = []

    for img_id, img in img_by_id.items():
        file_name = img["file_name"]   # e.g. "tb/tb0005.png"
        class_label = get_class_from_path(file_name)
        if class_label == -1:
            continue

        # Stem as image_id (without extension)
        stem = file_name.replace("/", "_").replace(".png", "")
        image_id = stem   # e.g. "tb_tb0005"

        # Concept labels (binary)
        concept_vec = [0, 0, 0]
        for ann in ann_by_imgid.get(img_id, []):
            cid = ann.get("category_id", -1)
            cidx = CAT_ID_TO_CONCEPT.get(cid, -1)
            if cidx >= 0:
                concept_vec[cidx] = 1

        row = {
            "image_id":    image_id,
            "class_label": class_label,
            "split":       split_name,
        }
        for j in range(3):
            row[f"concept_{j}"] = concept_vec[j]
        label_rows.append((file_name, row))

        # Bounding boxes — scale to TARGET_SIZE
        img_w = img["width"]
        img_h = img["height"]
        for ann in ann_by_imgid.get(img_id, []):
            cid = ann.get("category_id", -1)
            cidx = CAT_ID_TO_CONCEPT.get(cid, -1)
            if cidx < 0:
                continue
            bx, by, bw, bh = ann["bbox"]   # COCO: x,y,w,h in original image space
            x1 = int(bx / img_w * TARGET_SIZE)
            y1 = int(by / img_h * TARGET_SIZE)
            x2 = int((bx + bw) / img_w * TARGET_SIZE)
            y2 = int((by + bh) / img_h * TARGET_SIZE)
            bbox_rows.append({
                "image_id":    image_id,
                "concept_idx": cidx,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return label_rows, bbox_rows

print("Processing train split...")
train_label_rows, train_bbox_rows = process_split(train_imgs, train_anns, "train")
print("Processing val split...")
val_label_rows, val_bbox_rows = process_split(val_imgs, val_anns, "val")

# We use val as both 'val' and 'test' for ablation (no separate test labels in TBX11K)
# Also create bbox_eval split for val images that have bboxes
val_with_bbox = set(r["image_id"] for r in val_bbox_rows)
test_label_rows = []
bbox_eval_rows  = []
for file_name, row in val_label_rows:
    test_row = dict(row)
    test_row["split"] = "test"
    test_label_rows.append((file_name, test_row))
    if row["image_id"] in val_with_bbox:
        be_row = dict(row)
        be_row["split"] = "bbox_eval"
        bbox_eval_rows.append((file_name, be_row))

all_label_rows = train_label_rows + val_label_rows + test_label_rows + bbox_eval_rows
all_bbox_rows  = train_bbox_rows + val_bbox_rows

print(f"  Train: {len(train_label_rows)} images, {len(train_bbox_rows)} bboxes")
print(f"  Val:   {len(val_label_rows)} images,  {len(val_bbox_rows)} bboxes")
print(f"  Test (=val): {len(test_label_rows)} images")
print(f"  bbox_eval: {len(bbox_eval_rows)} images")

# ── Step 2: Symlink images ────────────────────────────────────────────────────
print("Creating image symlinks...")
imgs_dir = os.path.join(DEST, "images")
src_imgs_dir = os.path.join(SRC, "imgs")
created = 0
for file_name, row in set((fn, row["image_id"]) for fn, row in all_label_rows):
    src_path = os.path.join(src_imgs_dir, file_name)
    # Use image_id as filename stem
    image_id = row
    dst_path = os.path.join(imgs_dir, image_id + ".png")
    if not os.path.exists(dst_path) and os.path.exists(src_path):
        os.symlink(src_path, dst_path)
        created += 1
print(f"  Created {created} symlinks")

# ── Step 3: Write labels.csv ─────────────────────────────────────────────────
print("Writing labels.csv...")
fieldnames = ["image_id", "class_label", "split"] + [f"concept_{i}" for i in range(3)]
labels_path = os.path.join(DEST, "labels.csv")
with open(labels_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([row for _, row in all_label_rows])

# ── Step 4: Write bboxes.csv ─────────────────────────────────────────────────
print("Writing bboxes.csv...")
bboxes_path = os.path.join(DEST, "bboxes.csv")
with open(bboxes_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "concept_idx", "x1", "y1", "x2", "y2"])
    writer.writeheader()
    writer.writerows(all_bbox_rows)

print(f"\nDone.")
print(f"  labels.csv: {labels_path} ({len(all_label_rows)} rows)")
print(f"  bboxes.csv: {bboxes_path} ({len(all_bbox_rows)} rows)")
print(f"  images/: {len(list(Path(imgs_dir).iterdir()))} files")
