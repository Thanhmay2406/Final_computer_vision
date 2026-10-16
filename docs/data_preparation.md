# Dữ liệu 6 lớp dùng chung cho RF, MLP và CNN

Bộ dữ liệu đã được làm gọn, giữ nguyên 38.687 mẫu hợp lệ và split đã kiểm chứng. Ảnh được đổi đường dẫn/tên file, không resize hay mã hóa lại; 63 tọa độ landmark không được chuẩn hóa lại. Dữ liệu 12 lớp ngoài phạm vi và CSV nguồn đầy đủ đã được xóa theo lựa chọn của dự án.

## Cấu trúc hiện tại

```text
data/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
│       # Mỗi split chứa fist/, one/, peace/, three/, four/, palm/
│       # Mỗi ảnh có tên <UUID>.jpg
└── metadata/
    ├── manifest.csv
    ├── no_hand_log.csv
    └── preparation_report.json

assets/mediapipe/hand_landmarker.task
```

Ví dụ: `data/images/train/fist/000b9676-731b-4001-a658-d46b9389fc0e.jpg`. Model MediaPipe nằm ở `assets/mediapipe/`, độc lập với dữ liệu huấn luyện.

| Nhãn nguồn | Class ID | Thư mục lớp | Ý nghĩa |
|---|---:|---|---|
| `train_val_fist` | 0 | `fist` | Số 0 / búa |
| `train_val_one` | 1 | `one` | Số 1 |
| `train_val_peace` | 2 | `peace` | Số 2 / kéo |
| `train_val_three` | 3 | `three` | Số 3 |
| `train_val_four` | 4 | `four` | Số 4 |
| `train_val_palm` | 5 | `palm` | Số 5 / bao |

Số mẫu: **train 27.080 / validation 5.803 / test 5.804**. Log no-hand còn **3.747** bản ghi thuộc 6 lớp; không phải lớp thứ bảy và không thuộc split phân loại.

## Metadata và cách đọc

- `manifest.csv`: ID ổn định, nhãn nguồn, `class_id`, `gesture`, đường dẫn nguồn dùng truy vết, `image_path` cục bộ, SHA-256 ảnh, kích thước, split và `lm_0`–`lm_62`.
- `no_hand_log.csv`: đường dẫn nguồn và nhãn của các bản ghi không phát hiện được tay. Không có ảnh tương ứng để mở; giữ đường dẫn Kaggle làm lịch sử.
- `preparation_report.json`: mapping, seed, phiên bản thư viện, số lượng lớp/split, hash artifact và thông tin chuyển đổi. `input_sha256` cùng số lượng `all_*` là lịch sử của nguồn đầy đủ **đã bị xóa**; hash artifact trong `artifact_sha256` xác minh các file metadata hiện tại. `migration` ghi số ảnh giữ/xóa, hash model, fingerprints của nguồn cũ và trạng thái dọn nguồn.

`image_path` tương đối với **data/**, không phải `metadata/`. Chỉ `image_path` được dùng mở ảnh; `source_filepath` không phải đường dẫn cục bộ. `sample_id` và `source_label` giữ quy ước cũ để truy vết. Manifest sắp theo ID, dùng cột `split` để lấy tập.

```python
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

DATA_DIR = Path("data").resolve()
manifest = pd.read_csv(DATA_DIR / "metadata/manifest.csv", float_precision="round_trip")
train = manifest.loc[manifest["split"].eq("train")].copy()
columns = [f"lm_{i}" for i in range(63)]
X_train = train[columns].to_numpy(dtype=float)  # RF / MLP
landmarks = X_train.reshape(-1, 21, 3)
y_train = train["class_id"].to_numpy()
image_paths = [DATA_DIR / relative for relative in train["image_path"]]  # CNN
with Image.open(image_paths[0]) as source:
    rgb = ImageOps.exif_transpose(source).convert("RGB")
```

Ảnh được kiểm tra với EXIF orientation và RGB; loader CNN cần xử lý tương tự. Mọi scaler/PCA học từ dữ liệu phải fit trên train. Augmentation chỉ áp dụng train; dùng validation thử siêu tham số, giữ test cho đánh giá cuối cùng. Ba mô hình không tự chia lại tập.

## Kiểm tra hoặc xuất lại metadata

Từ thư mục gốc:

```bash
python scripts/prepare_data.py
```

Từ nơi khác:

```bash
python /path/to/Final_computer_vision/scripts/prepare_data.py
```

Script tự chọn cấu trúc canonical khi thấy `metadata/` hoặc `images/`, xác minh schema, mapping, hash metadata, toàn bộ ảnh và số lượng split. Nó giữ nguyên ID, tọa độ và split, xuất lại ba file metadata; không tạo `excluded_samples.csv` trong cấu trúc canonical. Nếu metadata/ảnh sai, dừng trước khi thay đầu ra.

Hỗ trợ `--data-dir`, `--output-dir`, `--seed`. Mặc định data cạnh thư mục scripts, đầu ra `<data-dir>/metadata`, seed 42. Seed phải khớp seed đã ghi trong báo cáo; không dùng cờ này để chia lại bộ canonical. Khi xuất metadata sang thư mục khác, `image_path` vẫn tương đối với data đầu vào. Đường dẫn cờ tương đối được hiểu theo thư mục đang đứng.

## Chuyển đổi từ cấu trúc nguồn cũ

```bash
# Chỉ kiểm tra, không thay đổi file
python scripts/reorganize_data.py

# Chuyển đổi và xóa dữ liệu dư đã liệt kê
python scripts/reorganize_data.py --apply
```

Có thể truyền `--data-dir` và `--assets-dir`; assets phải nằm ngoài data. Khi bộ canonical hoàn chỉnh đã tồn tại, script kiểm chứng bộ đích/model và không thay đổi. Khi đích không khớp hoặc chỉ có một phần cấu trúc, script từ chối ghi đè.

Lần chuyển đổi đầu dùng `prepared/manifest.csv` làm nguồn quyết định mẫu/split, đối chiếu hash CSV/manifest nguồn và giải mã ảnh. Toàn bộ bộ đích được sao chép vào staging, rồi kiểm tra hash, ảnh và mọi cột manifest ngoại trừ đường dẫn ảnh. Model được sao chép và kiểm chứng trước khi nguồn cũ bị xóa. Không thay đổi byte ảnh hay nhãn.

Chỉ sau các kiểm tra, script xóa `processed/`, hai CSV nguồn đầy đủ, bản model cũ và ba artifact cũ trong `prepared/`. File khác trong `prepared/` được giữ. Các nguồn dạng symlink bị từ chối để tránh xóa dữ liệu ngoài phạm vi. Fingerprints được kiểm tra lại trước khi xóa để phát hiện nguồn thay đổi trong lúc staging.

Báo cáo dùng trạng thái `published_cleanup_pending` trước khi dọn nguồn và `complete` sau khi hoàn tất. Nếu bị gián đoạn, chạy lại kiểm chứng đích, model và các nguồn còn tồn tại trước khi dọn. Nguồn không khớp fingerprints sẽ ngăn xóa. Nếu một thư mục nguồn đã bị xóa một phần và fingerprint không còn khớp, script dừng để kiểm tra thủ công, không tự xóa phần còn lại. Không chạy đồng thời nhiều tiến trình sửa cùng data.

Script chuẩn bị vẫn hỗ trợ **nguồn cũ** cho dữ liệu nhập mới và kiểm thử: `landmarks.csv`, `no_hand_log.csv`, `processed/<nhãn nguồn>/`. Trong chế độ này, nó kiểm tra 63 cột, ghép tên ảnh đầy đủ, giải mã ảnh, lọc lỗi/trùng/xung đột và tạo split stratified hai bước 70/15/15 với seed cố định; đầu ra mặc định `prepared/` gồm manifest, exclusions và report. Nguồn cũ không tồn tại nữa trong data của dự án này.

## Kiểm chứng và giới hạn

```bash
python -m pytest -q
```

Các kiểm thử dùng ảnh/CSV nhỏ thật, bao phủ chuẩn bị nguồn cũ, chuyển đổi, tính tái lập, bảo toàn dữ liệu, lỗi staging, ảnh thiếu/hash sai, đường dẫn đích trùng, đích đã tồn tại, tiếp tục dọn nguồn bị sửa và symlink. Notebook `notebooks/EDA.ipynb` đọc metadata mới và đường dẫn ảnh trực tiếp.

Split ngẫu nhiên hiện tại **chưa bảo đảm tách người chụp/phiên quay**. Chưa có mã nguồn xác nhận chuẩn hóa landmark và crop ảnh. SHA-256 chỉ phát hiện file giống byte, không bảo đảm phát hiện mọi ảnh gần giống hoặc cùng người/nền. Khi có metadata người chụp, cần protocol theo nhóm trước khi kết luận khả năng tổng quát hóa. Bộ hiện tại phục vụ so sánh bộ phân loại trên mẫu có cả landmark và ảnh hợp lệ; không đại diện cho chất lượng end-to-end hoặc FPS trên webcam.
