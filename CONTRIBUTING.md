# 🤝 Hướng dẫn đóng góp — Contributing Guide

> Tài liệu này hướng dẫn cách đóng góp vào dự án **FireDetection_Project** một cách chuyên nghiệp, nhất quán và dễ theo dõi, tuân theo chuẩn [**Conventional Commits**](https://www.conventionalcommits.org/).

---

## 📋 Mục lục

- [Conventional Commits Format](#-conventional-commits-format)
- [Danh sách Type](#-danh-sách-type)
- [Danh sách Scope](#-danh-sách-scope)
- [Ví dụ thực tế](#-ví-dụ-commit-thực-tế)
- [Chiến lược branch](#-chiến-lược-branch)
- [Quy trình đóng góp](#-quy-trình-đóng-góp)
- [Quy tắc chung](#-quy-tắc-chung)

---

## 📐 Conventional Commits Format

Mỗi commit message phải tuân theo cấu trúc sau:

```
<type>(<scope>): <mô tả ngắn gọn>

[body - tuỳ chọn]

[footer - tuỳ chọn]
```

| Phần | Bắt buộc | Mô tả |
|---|:---:|---|
| `type` | ✅ | Loại thay đổi (xem danh sách bên dưới) |
| `scope` | ✅ | Phạm vi ảnh hưởng (module/thư mục) |
| `mô tả` | ✅ | Tóm tắt ngắn gọn, **tối đa 72 ký tự**, dùng tiếng Việt hoặc tiếng Anh |
| `body` | ❌ | Giải thích chi tiết lý do và nội dung thay đổi |
| `footer` | ❌ | Tham chiếu issue, breaking change, co-author |

### 📌 Ví dụ đầy đủ

```
feat(model): thêm RT-DETR-L với backbone ResNet-50

- Tích hợp RT-DETR-L từ Ultralytics
- Cấu hình num_classes=2 (Fire, Smoke)
- Hỗ trợ inference ở độ phân giải 640x640

Closes #12
```

---

## 🏷️ Danh sách Type

| Type | Ý nghĩa | Khi nào dùng |
|---|---|---|
| `feat` | ✨ Thêm tính năng mới | Thêm model mới, API endpoint, tính năng cảnh báo |
| `fix` | 🐛 Sửa lỗi | Sửa bug phát hiện sai, lỗi inference, lỗi alert |
| `refactor` | ♻️ Cải thiện code | Tái cấu trúc không thay đổi chức năng |
| `docs` | 📝 Tài liệu | Cập nhật README, thêm docstring, hướng dẫn |
| `data` | 🗃️ Thay đổi dữ liệu | Thêm/xóa dataset, cập nhật annotation, augmentation |
| `chore` | 🔧 Cấu hình, dependencies | Cập nhật `requirements.txt`, `.gitignore`, configs |
| `test` | 🧪 Test | Thêm/sửa unit test, integration test |
| `ci` | 🚀 CI/CD, Docker | Cập nhật Dockerfile, GitHub Actions workflow |
| `notebook` | 📓 Jupyter Notebook | Thêm/sửa notebook huấn luyện, phân tích |
| `style` | 💄 Format code | Sửa format, khoảng trắng, import order (không thay đổi logic) |

---

## 🎯 Danh sách Scope

Sử dụng scope phù hợp với phần của dự án bị ảnh hưởng:

| Scope | Thư mục / Module | Mô tả |
|---|---|---|
| `model` | `src/models/` | RT-DETR model, wrapper, inference |
| `src` | `src/` | Source code chung |
| `web` | `web/` | FastAPI server, giao diện web |
| `data` | `data/`, `src/data/` | Dataset, preprocessing, augmentation |
| `configs` | `configs/` | File cấu hình YAML |
| `notebooks` | `notebooks/` | Jupyter notebooks |
| `docker` | `Dockerfile` | Docker configuration |
| `api` | `web/main.py` | API endpoints |
| `deps` | `requirements.txt` | Dependencies, packages |

> 💡 **Tip:** Nếu thay đổi ảnh hưởng nhiều scope, hãy chọn scope **chính** hoặc dùng `src` cho thay đổi tổng quát.

---

## 💡 Ví dụ Commit Thực tế

Dưới đây là các ví dụ commit thực tế cho dự án **Fire Detection**:

### ✨ feat — Thêm tính năng mới

```bash
feat(model): thêm SAHI slicing inference cho vật thể nhỏ
feat(web): thêm endpoint /predict nhận ảnh base64
feat(api): tích hợp cảnh báo qua Telegram Bot
feat(data): thêm pipeline augmentation với Albumentations
feat(model): hỗ trợ batch inference cho video stream
```

### 🐛 fix — Sửa lỗi

```bash
fix(model): sửa lỗi false positive với đèn LED đỏ
fix(web): sửa lỗi CORS khi gọi API từ frontend
fix(data): sửa lỗi label bị lệch sau khi resize ảnh
fix(api): sửa lỗi timeout khi gửi cảnh báo Zalo OA
fix(model): sửa ngưỡng confidence threshold cho lớp Smoke
```

### ♻️ refactor — Cải thiện code

```bash
refactor(src): tách logic alert thành module riêng utils/alert.py
refactor(model): tối ưu pipeline inference giảm latency 30%
refactor(data): chuẩn hoá cấu trúc FireSmokeDataset
refactor(web): chuyển sang async/await cho các API call
```

### 📝 docs — Tài liệu

```bash
docs(readme): cập nhật hướng dẫn cài đặt môi trường CUDA
docs(src): thêm docstring cho class Trainer và Evaluator
docs(data): thêm README mô tả cấu trúc 5 thư mục dataset
docs(configs): thêm chú thích cho các tham số trong default.yaml
```

### 🗃️ data — Thay đổi dữ liệu

```bash
data(data): thêm 500 ảnh hẻm TPHCM vào 02_Alley_Context
data(data): bổ sung hard negative samples từ khói bếp, đèn quảng cáo
data(data): cập nhật annotation YOLO format cho tập 04_SAHI
data(data): xoá ảnh trùng lặp bằng imagehash deduplication
```

### 🔧 chore — Cấu hình, dependencies

```bash
chore(deps): nâng cấp ultralytics lên phiên bản 8.2.0
chore(configs): cập nhật default.yaml thêm cấu hình SAHI
chore(deps): thêm python-telegram-bot==20.7 vào requirements.txt
chore(src): thêm .gitignore cho thư mục runs/ và weights/
```

### 🚀 ci — CI/CD, Docker

```bash
ci(docker): tối ưu Dockerfile dùng multi-stage build
ci(docker): thêm HEALTHCHECK cho container
ci(docker): cập nhật base image sang pytorch/pytorch:2.0.1-cuda11.7
```

### 📓 notebook — Jupyter Notebook

```bash
notebook(notebooks): thêm notebook phân tích kết quả Stage 1
notebook(notebooks): cập nhật pipeline huấn luyện 3 giai đoạn
notebook(notebooks): thêm visualization confusion matrix và mAP curves
```

### 🧪 test — Testing

```bash
test(model): thêm unit test cho FireDetectionModel.predict()
test(data): thêm test kiểm tra tính hợp lệ của annotation YOLO
test(api): thêm integration test cho endpoint /predict
```

### 💄 style — Format code

```bash
style(src): format code theo PEP8 với black formatter
style(web): sắp xếp lại thứ tự import trong main.py
```

---

## 🌿 Chiến lược Branch

```
main
 │  ← Production: chỉ merge từ develop sau khi kiểm tra đầy đủ
 │
develop
 │  ← Nhánh tích hợp chính: mọi tính năng merge vào đây trước
 │
 ├── feat/<tên-tính-năng>     ← Phát triển tính năng mới
 │   Ví dụ: feat/sahi-integration
 │           feat/telegram-alert
 │           feat/web-dashboard
 │
 ├── fix/<mô-tả-lỗi>          ← Sửa lỗi
 │   Ví dụ: fix/false-positive-led
 │           fix/cors-api
 │
 └── data/<mô-tả-dataset>     ← Thêm/cập nhật dataset
     Ví dụ: data/add-alley-images
             data/hard-negative-samples
```

### Quy tắc đặt tên branch

| Loại | Pattern | Ví dụ |
|---|---|---|
| Tính năng mới | `feat/<tên>` | `feat/sahi-integration` |
| Sửa lỗi | `fix/<mô-tả>` | `fix/false-positive-led` |
| Dataset | `data/<mô-tả>` | `data/add-alley-context` |
| Tài liệu | `docs/<mô-tả>` | `docs/update-readme` |
| Cấu hình | `chore/<mô-tả>` | `chore/upgrade-ultralytics` |

---

## 🔄 Quy trình đóng góp

```
1. Fork / Clone repository
        │
        ▼
2. Tạo branch mới từ develop
   git checkout develop
   git checkout -b feat/ten-tinh-nang
        │
        ▼
3. Thực hiện thay đổi
   (code, test, docs)
        │
        ▼
4. Commit theo Conventional Commits
   git add .
   git commit -m "feat(model): thêm SAHI inference"
        │
        ▼
5. Push branch lên remote
   git push origin feat/ten-tinh-nang
        │
        ▼
6. Tạo Pull Request → develop
   (điền đầy đủ mô tả, checklist)
        │
        ▼
7. Review & Merge
   (cần ít nhất 1 người review)
```

### Checklist Pull Request

Khi tạo Pull Request, hãy đảm bảo:

- [ ] Commit message tuân theo Conventional Commits
- [ ] Code được test cục bộ trước khi tạo PR
- [ ] Không có conflict với nhánh `develop`
- [ ] Đã cập nhật tài liệu liên quan (nếu có)
- [ ] Không commit file nhạy cảm (API key, password, model weights lớn)

---

## 📏 Quy tắc chung

### ✅ Nên làm

- ✅ Viết commit message **rõ ràng, ngắn gọn** (tối đa 72 ký tự dòng đầu)
- ✅ Mỗi commit chỉ giải quyết **một việc cụ thể**
- ✅ Tạo **Pull Request** và yêu cầu review trước khi merge vào `develop`
- ✅ Dùng tiếng Việt hoặc tiếng Anh nhất quán trong một commit
- ✅ Tham chiếu issue liên quan trong footer: `Closes #12`, `Refs #8`

### ❌ Không nên làm

- ❌ **Không commit trực tiếp lên `main`** — luôn tạo PR
- ❌ Không dùng commit message chung chung như `fix bug`, `update`, `wip`
- ❌ Không commit file nhạy cảm: API keys, `.env`, model weights (> 100MB)
- ❌ Không gộp nhiều thay đổi không liên quan vào một commit
- ❌ Không force push lên `develop` hoặc `main`

### 📏 Quy tắc độ dài

```
feat(model): thêm SAHI slicing inference cho ảnh độ phân giải cao  ← tối đa 72 ký tự
             |___________________________________________________|
                          dòng đầu (subject line)

Body: giải thích chi tiết nếu cần (mỗi dòng ≤ 72 ký tự)
Footer: Closes #12, BREAKING CHANGE: ...
```

---

## 🚫 File không nên commit

Các file sau đây đã được cấu hình trong `.gitignore` và **không được commit**:

```
# Model weights
*.pt, *.pth, *.onnx

# Dataset (dữ liệu lớn)
data/*/images/, data/*/labels/

# Môi trường
venv/, .env, __pycache__/

# Kết quả training
runs/, outputs/, wandb/

# Secrets
*.key, *_secret*, credentials*
```

---

## 📞 Liên hệ

Nếu có câu hỏi về quy trình đóng góp, vui lòng liên hệ:

| Tên | Email |
|---|---|
| Vo Xuan Quang | 523H0173@student.tdtu.edu.vn |
| Hoang Xuan Thanh | 523H0178@student.tdtu.edu.vn |

---

<div align="center">
  <sub>🔥 FireDetection_Project — Ton Duc Thang University (TDTU)</sub>
</div>
