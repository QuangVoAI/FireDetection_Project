# 📁 Hướng dẫn Chuẩn bị Dataset

## Cấu trúc thư mục

Tạo 5 thư mục con, mỗi thư mục có `images/` và `labels/`:

```
data/
├── 01_Positive_Standard/
│   ├── images/     ← Đặt ảnh lửa/khói rõ ràng ở đây
│   └── labels/     ← File .txt labels (YOLO format)
├── 02_Alley_Context/
│   ├── images/     ← Ảnh hẻm TPHCM
│   └── labels/
├── 03_Negative_Hard_Samples/
│   ├── images/     ← Ảnh khó (hơi phở, đèn đỏ, etc.)
│   └── labels/     ← File rỗng (không có lửa/khói)
├── 04_SAHI_Small_Objects/
│   ├── images/     ← Ảnh vật thể nhỏ/xa
│   └── labels/
└── 05_Ambient_Context_Null/
    ├── images/     ← Ảnh bối cảnh bình thường (không có lửa/khói)
    └── labels/
```

## Annotation Format (YOLO)

Mỗi ảnh có 1 file `.txt` cùng tên trong thư mục `labels/`:

```
# <class_id> <cx> <cy> <width> <height>
# Tất cả giá trị normalized (0-1)
0 0.512 0.334 0.245 0.312   # Fire (Lửa)
1 0.701 0.221 0.190 0.280   # Smoke (Khói)
```

**Class mapping:**
- `0` = Fire (Lửa) 🔥
- `1` = Smoke (Khói) 💨

## Nguồn dữ liệu gợi ý

| Folder | Nguồn |
|---|---|
| `01_Positive_Standard` | [COCO Fire](https://huggingface.co/datasets), [Kaggle Fire](https://www.kaggle.com/datasets?search=fire+detection), [Roboflow](https://universe.roboflow.com/) |
| `02_Alley_Context` | Tự chụp/quay tại hẻm TPHCM, Google Street View |
| `03_Negative_Hard_Samples` | Tự chụp: quán phở, xe máy ban đêm, đèn LED đỏ |
| `04_SAHI_Small_Objects` | Crop/augment từ folder 01, camera ban công |
| `05_Ambient_Context_Null` | Rừng núi/đường phố/nhà cửa bình thường (không có lửa/khói) |

## Tool gán nhãn (Annotation)

Recommend: [LabelImg](https://github.com/HumanSignal/labelImg) hoặc [Roboflow](https://roboflow.com/)

```bash
# Cài LabelImg
pip install labelImg
labelImg data/01_Positive_Standard/images/
# Chon format YOLO → vẽ bbox → save
```
