# Nhận diện sáu cử chỉ tay bằng MediaPipe và học máy

Pipeline so sánh landmark + Random Forest, landmark + MLP và ảnh RGB + CNN. Sáu lớp thị giác: `fist`, `one`, `peace`, `three`, `four`, `palm`. Quy ước ứng dụng: số 0–5; `fist` = búa, `peace` = kéo, `palm` = bao. Demo có hai chế độ hiển thị riêng: số đếm và kéo–búa–bao; chưa có logic trò chơi hoặc chuyển nhãn thành lệnh điều khiển.

## Kết quả hiện hành

Phiên full **manual_002** đã train trên Kaggle GPU T4 và được tải/kiểm chứng local. Dùng các đường dẫn sau:

| Thành phần | Đường dẫn |
|---|---|
| Source dùng chung | [handgesture/](handgesture/) |
| Notebook Kaggle chính, output trống | [07_interactive_workflow.ipynb](notebooks/kaggle_v2/07_interactive_workflow.ipynb) |
| Notebook full đã chạy, có output | [full_manual_002.executed.ipynb](results_v2/kaggle_full_manual_002/notebook/full_manual_002.executed.ipynb) |
| Dữ liệu khớp checkpoint full | `data_v2/` |
| Checkpoint và kết quả full | `results_v2/kaggle_full_manual_002/experiment/final/` |
| Báo cáo chất lượng | [comparison/report.md](results_v2/kaggle_full_manual_002/experiment/final/comparison/report.md) |
| So sánh có latency CPU | [benchmark comparison/report.md](results_v2/kaggle_full_manual_002/benchmarks/cpu/classifier/comparison/report.md) |
| Kiểm tra đối chiếu đề cương | [docs/proposal_audit.md](docs/proposal_audit.md) |

Test Accuracy: RF **91,63%**, MLP **92,83%**, CNN được chọn bằng validation (MobileNetV3-Small) **95,57%**, trên cùng 3.680 mẫu test đã qua detector. Đây là metric classifier offline; chưa phải Accuracy/FPS webcam. Chưa chốt model triển khai vì thiếu benchmark end-to-end và nghiệm thu webcam.

`data/` và `results/` giữ baseline lịch sử. `data_v2_local_archive/` giữ nguyên bản preprocessing local cũ, có manifest khác full Kaggle; không dùng nó với checkpoint full. Đường dẫn `results_v2/kaggle_full_manual_002/dataset` là symlink tương thích trỏ về `data_v2/`. [Bản đồ source và dữ liệu](docs/source_map.md) mô tả đầy đủ vai trò các thư mục.

## Nghiệm thu hiện tại

Đã chạy ba lượt webcam thật ở640×480,600 frame/model: khoảng16,65FPS, có log recognized/uncertain/no_hand và đóng camera bình thường. Đây là ba lượt khác đầu vào; chưa thay thế benchmark chung hoặc xác nhận đúng/sai sáu cử chỉ. Người dùng chọn chưa ghi video.

- [Báo cáo hệ thống và thực nghiệm](docs/reports/full_manual_002.md)
- [Checklist và quy trình nghiệm thu](docs/acceptance/webcam_protocol.md)
- [Kết quả live và timing](results_v2/kaggle_full_manual_002/acceptance/report.md)

## Cài đặt local

Dùng Python 3.11 và môi trường riêng:

```bash
python3.11 -m venv .venv-v2
.venv-v2/bin/python -m pip install -r requirements/local.txt
.venv-v2/bin/python -m pip install -r requirements/local-torch.txt
```

Môi trường đã kiểm thử và dependency cloud nằm trong [requirements/](requirements/). Mỗi phiên lưu môi trường thực tế. Detector: `assets/mediapipe/hand_landmarker.task`; pretrained MobileNet: `assets/pretrained/mobilenet_v3_small-047dcff4.pth`.

## Kiểm chứng và benchmark kết quả full

```bash
.venv-v2/bin/python scripts/train_v2.py verify \
  --results-root results_v2/kaggle_full_manual_002/experiment/final --data-root data_v2
```

Benchmark classifier CPU đã có tại `results_v2/kaggle_full_manual_002/benchmarks/cpu/classifier/`: một thread, batch 1, warm-up 20, 200 mẫu test cố định, median/p95. Để tạo lượt đo mới:

```bash
.venv-v2/bin/python scripts/benchmark_v2.py \
  --results-root results_v2/kaggle_full_manual_002/experiment/final \
  --data-root data_v2 --output-dir results_v2/benchmark-new
```

Để bổ sung benchmark end-to-end, thêm `--video /path/to/hand-video.mp4 --display`. Cần cùng một video ít nhất 330 frame cho ba model: 30 warm-up + 300 frame đo. Nhóm hiện chưa có video này. Không suy FPS webcam từ latency classifier.

## Chạy webcam với checkpoint full

```bash
.venv-v2/bin/python scripts/webcam_demo.py --model mlp --mode count \
  --checkpoint-dir results_v2/kaggle_full_manual_002/experiment/final \
  --camera 0 --width 640 --height 480
```

Đổi `--model rf|mlp|cnn` để thử cả ba. Chọn `--mode count` (mặc định) để chỉ hiện số0–5, hoặc `--mode rps` để chỉ hiện Búa/Kéo/Bao. Phím `m` đổi chế độ ngay khi đang chạy; tên chế độ hiện trên overlay. Ở chế độ rps, tư thế1/3/4 hiện “Ngoài bộ cử chỉ”. Đây là ánh xạ của cùng model sáu lớp, không phải model ba lớp mới. CSV giữ `label` gốc và thêm `display_mode`/`display_label` của frame thực sự đã hiển thị; summary ghi chế độ đầu/cuối và số lần đổi. `mlp` trong ví dụ là lựa chọn để chạy thử, chưa phải model triển khai được chốt. `q` hoặc Esc để thoát. Confidence mặc định 0,7; cửa sổ 7 frame, cần 5 phiếu cùng nhãn. Mất tay xóa nhãn ngay; chỉ lật hình hiển thị. Công cụ lưu timing CSV/protocol, không tự lưu ảnh/video. Phải kiểm tra thủ công sáu cử chỉ, mất tay, confidence thấp và đóng camera.

Chỉ dùng `--deployment /path/to/deployment.json` sau benchmark đủ frame/có tay và đạt mục tiêu 15 FPS. Xem [pipeline v2](docs/pipeline_v2.md) để biết chi tiết protocol.

## Chạy thực nghiệm mới trên Kaggle

Import [07_interactive_workflow.ipynb](notebooks/kaggle_v2/07_interactive_workflow.ipynb), bật GPU và gắn hai Input:

- `thanhmay2406/hand-gesture`: dữ liệu nguồn/split.
- `thanhmay2406/hand-gesture-v2-runtime-assets-20261015`: detector và pretrained weights.

Notebook mặc định smoke. Để full, đặt `SMOKE_TEST=False`, `RUN_ID` mới và `EVALUATE_TEST=False`. Chạy các cell dependency/GPU → Input → preprocessing → RF/MLP/ba CNN → validation. Sau khi chốt CNN, chỉ đổi `EVALUATE_TEST=True`, chạy lại cell cấu hình rồi test/export; không train lại hoặc dùng test để chọn cấu hình.

Kernel cần Python3.11/3.12 và MediaPipe0.10.21. Với image Python3.13+, bật `PREPARE_COMPATIBLE_KERNEL=True`, chạy setup tới `NATIVE_READY`; bật `SWITCH_COMPATIBLE_KERNEL=True`, chạy cell chuyển kernel rồi chạy lại từ cấu hình sau khi kết nối lại. Restart Session có thể quay về Python của image. Giữ `NUM_WORKERS=0` và `REQUIRE_CUDA=True` trên Kaggle. Sau khi chạy thủ công xong, Quick Save kèm output. Không dùng Save & Run All để trông chờ tự xử lý bước chuyển kernel tương tác.

Notebook phân phối sinh từ cùng source local và không chứa output. Các notebook tách bước và công cụ API/watcher vẫn được giữ để tái lập; xem [notebooks/README.md](notebooks/README.md), [scripts/README.md](scripts/README.md). File trạng thái watcher cũ không phản ánh phiên tương tác full hiện tại.

## Dữ liệu và giới hạn thực nghiệm

Dữ liệu chính dùng Kaggle theo điều chỉnh đã thống nhất: 38.687 mẫu nguồn → 24.755 được giữ, 13.932 bị loại (đều `no_hand`). Split dẫn xuất: train 17.385, validation 3.690, test 3.680, giữ ID/split nguồn. Không tự chia lại hoặc trộn manifest local cũ. CNN dùng ROI; RF/MLP dùng 63 đặc trưng chuẩn hóa theo cổ tay và kích thước tay. Không có metadata người chụp/phiên quay; kết quả chưa chứng minh tổng quát hóa sang người mới.

Chuẩn bị lại dữ liệu vào thư mục mới nếu cần thử nghiệm, không ghi đè bản đang khớp checkpoint:

```bash
.venv-v2/bin/python scripts/prepare_data_v2.py \
  --source-root data --output-root data_v2-new --workers 0
```

Không dùng `scripts/normalize_results.py` cho output v2; đó là công cụ baseline cũ. Không tính cải thiện trực tiếp so với baseline lịch sử khi tập đánh giá/preprocessing khác nhau.

## Kiểm thử và sinh lại notebook

```bash
.venv-v2/bin/python -m pytest -q
.venv-v2/bin/python scripts/build_kaggle_notebooks.py
.venv-v2/bin/python scripts/build_v2_notebooks.py
.venv-v2/bin/python scripts/smoke_test_v2.py
```

Test/smoke không thay thế GPU full training hoặc webcam nghiệm thu. Lịch sử triển khai ở [docs/implementation/progress.md](docs/implementation/progress.md); lần dọn repo này có danh sách thay đổi tại [repository_cleanup.json](docs/implementation/repository_cleanup.json).
