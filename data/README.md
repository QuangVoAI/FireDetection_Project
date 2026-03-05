# Data Directories

This directory contains all raw and processed images used to train and evaluate the fire detection model.

## Raw Data Sub-folders

| Folder | Description |
|--------|-------------|
| `01_Positive_Standard/` | Clear fire and smoke images (black smoke, white smoke, visible flames) |
| `02_Alley_Context/` | Real-world alley / residential contexts specific to Ho Chi Minh City (District 7, District 4, Tan Binh, etc.) |
| `03_Negative_Hard_Samples/` | Confusing objects: pho steam, water vapour, motorbike tail lights, LED signs, red clothing |
| `04_SAHI_Small_Objects/` | High-resolution images with small/distant fire & smoke (balcony view deep into alleys); use SAHI inference |
| `05_Real_Situation/` | Real fire incident photos from Vietnamese news (VTV, VnExpress) |

## Annotation Format

- **Tool**: Roboflow or LabelImg
- **Format**: YOLO (.txt) – one file per image, each line: `<class_id> <cx> <cy> <w> <h>` (normalised 0–1)
- **Classes**: `0 = Fire`, `1 = Smoke`
- Place label files in a `labels/` sibling directory next to each `images/` directory.

## Processed Data

After running `python scripts/preprocess.py`, preprocessed images and labels will be written to:

```
data/processed/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Default split ratios: **80% train / 10% val / 10% test**.

## Data Sources

- [Kaggle fire/smoke datasets](https://www.kaggle.com/)
- Google Images search
- Vietnamese news portals: VTV, VnExpress

> **Note**: Raw images are excluded from version control (see `.gitignore`).
> Only `.gitkeep` placeholder files are tracked to preserve the directory structure.
