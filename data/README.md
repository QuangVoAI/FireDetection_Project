# Dataset Setup Guide — Hệ thống Phát hiện Cháy Sớm

This directory contains the training and validation data for the Early Fire Detection System.

## Required Folder Structure

Each of the 5 data folders must follow YOLO-format structure:

```
data/
├── 01_Positive_Standard/
│   ├── images/      ← JPEG/PNG image files
│   └── labels/      ← YOLO .txt annotation files
│
├── 02_Alley_Context/
│   ├── images/
│   └── labels/
│
├── 03_Negative_Hard_Samples/
│   ├── images/
│   └── labels/
│
├── 04_SAHI_Small_Objects/
│   ├── images/
│   └── labels/
│
└── 05_Real_Situation/
    ├── images/
    └── labels/
```

## Annotation Format (YOLO)

Each `.txt` label file corresponds to one image (same filename, different extension).

**Format per line:**
```
<class_id> <cx> <cy> <width> <height>
```

All values are **normalised to [0, 1]** relative to the image dimensions.

**Class mapping:**
| Class ID | Class Name | Vietnamese |
|----------|------------|------------|
| `0`      | Fire       | Lửa        |
| `1`      | Smoke      | Khói       |

**Example** (`labels/fire_001.txt`):
```
0 0.512 0.334 0.245 0.312
1 0.701 0.221 0.190 0.280
```
Line 1: Fire at centre (51.2%, 33.4%) with size 24.5% × 31.2% of image.
Line 2: Smoke at centre (70.1%, 22.1%) with size 19.0% × 28.0% of image.

## Folder Descriptions

### `01_Positive_Standard/`
Clear fire and smoke images under standard conditions. Sources:
- Open datasets (Kaggle Fire Detection, COCO subsets)
- Publicly available fire/smoke image collections

### `02_Alley_Context/`
Real alley (hẻm) scenes from Ho Chi Minh City: Quận 7, Quận 4, Tân Bình.
- Street camera footage / photos
- Manually collected field images

### `03_Negative_Hard_Samples/`
Hard negative samples that resemble fire/smoke (potential false positives):
- Steam/smoke from phở (noodle soup) vendors
- Water vapor / cooking steam
- Motorbike tail lights at night
- Red LED advertising signs (đèn quảng cáo)
- Red clothing, motorbike seat covers
- Sunset reflections on windows

### `04_SAHI_Small_Objects/`
Images featuring small or distant fire/smoke objects:
- Balcony/rooftop camera angles
- Fires viewed from 30+ metres away
- Partially occluded smoke in narrow alleyways
- High-resolution images where fires appear as tiny spots

### `05_Real_Situation/`
Real fire incident images from Vietnamese news sources:
- VTV (Vietnam Television)
- VnExpress
- Tuổi Trẻ newspaper
- Thanh Niên newspaper

> ⚠️ Ensure you have the right to use and redistribute any images you include.

## Data Statistics (target)

| Folder | Min Images | Notes |
|--------|------------|-------|
| 01_Positive_Standard | 1000 | Balanced Fire/Smoke |
| 02_Alley_Context | 500 | HCMC-specific scenes |
| 03_Negative_Hard_Samples | 800 | Roughly equal to positives |
| 04_SAHI_Small_Objects | 400 | High-resolution preferred |
| 05_Real_Situation | 200 | Curated quality over quantity |

## Validation

Run the annotation validator before training:

```python
from src.data.preprocessing import validate_annotations

for folder in ["data/01_Positive_Standard", "data/02_Alley_Context"]:
    valid, invalid, errors = validate_annotations(folder)
    print(f"{folder}: {valid} valid, {invalid} invalid")
    for e in errors[:5]:
        print("  ", e)
```

## Deduplication

Remove duplicate images using perceptual hash:

```python
from src.data.preprocessing import deduplicate_dataset

removed = deduplicate_dataset("data/01_Positive_Standard/images")
print(f"Removed {len(removed)} duplicates")
```
