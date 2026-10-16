# Bản đồ source và dữ liệu

Code đã tách theo trách nhiệm; giữ đường dẫn module/script để notebook generator, import và bộ test tiếp tục dùng cùng source. Lần tổ chức này không đổi thuật toán, checkpoint hoặc measured output.

## Source chính

| Vị trí | Trách nhiệm |
|---|---|
| `handgesture/preprocessing.py` | Detector IMAGE/VIDEO, 63 đặc trưng, ROI, fingerprint |
| `handgesture/data.py` | Dataset/manifest, ID/split/hash, journal và mẫu bị loại |
| `handgesture/models.py` | RF/MLP/CNN GAP/spatial/MobileNet và predictor |
| `handgesture/training.py` | Train/validation/chọn CNN/finalize/verify/report |
| `handgesture/live.py` | Webcam/video loop, làm mượt, trạng thái và render/timing |
| `handgesture/benchmark.py` | Classifier CPU, cùng video ba model, chốt deployment |
| `handgesture/runtime.py` | Stage, progress nguyên tử, CUDA check, wrapper deadline |
| `handgesture/kernel_bootstrap.py` | Chuẩn bị và chuyển native kernel Kaggle tương thích |
| `scripts/` | CLI, generator, smoke và công cụ Kaggle; xem README trong thư mục |
| `tests/` | Kiểm tra geometry, dataset, model/output, notebook, runtime/watcher, webcam/benchmark |
| `requirements/` | Môi trường local, Kaggle, CLI tools và lock đã kiểm thử |

## Dữ liệu và output

| Vị trí | Trạng thái và cách dùng |
|---|---|
| `data/` | Dữ liệu nguồn đã chuẩn bị; giữ nguyên baseline/source split |
| `data_v2/` | Dữ liệu chuẩn từ full Kaggle manual_002; dùng với checkpoint full |
| `data_v2_local_archive/` | Bản local cũ giữ để truy vết; cùng ID/split nhưng khác manifest và một số ROI/landmark |
| `results/` | Baseline lịch sử, không trộn vào báo cáo full v2 |
| `results_v2/kaggle_full_manual_002/experiment/` | Năm lượt train và ba model final, raw metrics/predictions/history được giữ nguyên |
| `results_v2/kaggle_full_manual_002/notebook/` | Notebook thực thi và source tải từ Kaggle |
| `results_v2/kaggle_full_manual_002/benchmarks/` | Benchmark CPU mới; tách riêng measured artifacts của Kaggle |
| `results_v2/kaggle_full_manual_002/dataset` | Symlink `../../data_v2`; giữ tương thích đường dẫn tải xuống cũ |
| `results_v2/kaggle_full_manual_002/downloads/` | Giữ results_v2.zip và manifest gốc; data_v2.zip đã bỏ sau kiểm tra byte từng member với dữ liệu giải nén |
| `assets/` | Detector và weights pretrained, không xóa khi dọn output |

Không sửa đường dẫn hoặc nội dung bên trong `run.json`, `metrics.json`, `selection.json` chỉ để thay chuỗi `/kaggle/working`: đó là provenance của phiên train. CLI nhận đường dẫn local riêng.

Đã xóa staging preprocessing dở dang, thư mục smoke-download rỗng, cache có thể sinh lại và ZIP dữ liệu dư thừa. Danh sách/byte ở hồ sơ `docs/implementation/repository_cleanup.json` giữ local. Chỉ các thư mục dữ liệu dẫn xuất được chuyển; `data/`, `results/`, checkpoint, dự đoán và notebook phân phối được giữ nguyên.

## Tài liệu

- [Đối chiếu đề cương](proposal_audit.md): phần đạt, thiếu và thay đổi phạm vi đã thống nhất.
- [Pipeline v2](pipeline_v2.md): protocol kỹ thuật.
- [Dữ liệu baseline](data_preparation.md) và [Kaggle baseline](kaggle_training.md): lịch sử, không phải hướng dẫn chọn model full hiện hành.
- Nhật ký `docs/implementation/` (giữ local): trạng thái hiện hành và cách đọc hồ sơ cũ.

## Bản GitHub chia sẻ

Trong bản clone mới, model nằm tại `models/manual_002/` sau khi tải Release, số liệu tại `reports/manual_002/`. Các thư mục data/results/results_v2 và archive ở trên mô tả workspace tác giả, không nằm trong Git. Xem README gốc để có lệnh chạy cho máy bạn.
