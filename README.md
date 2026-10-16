# HandGesture — thực nghiệm webcam và báo cáo

So sánh **landmark + Random Forest**, **landmark + MLP** và **ảnh RGB + CNN** cho sáu cử chỉ tay tĩnh. Repo cung cấp source, notebook Kaggle, checkpoint full qua GitHub Release và số liệu đã đo để nhóm chạy webcam, phân tích kết quả và viết báo cáo.

- **Chạy webcam:** không cần tải dataset, không cần tài khoản Kaggle, suy luận trên CPU.
- **Viết báo cáo:** xem [số liệu manual_002](reports/manual_002/README.md) và [báo cáo thực nghiệm](docs/reports/full_manual_002.md).
- **Train/thực nghiệm lại:** dùng [notebook Kaggle chính](notebooks/kaggle_v2/07_interactive_workflow.ipynb) và [protocol v2](docs/pipeline_v2.md).

## 1. Clone và cài môi trường

Môi trường đã kiểm thử: **Python 3.11, Linux, CPU**, sklearn 1.6.1 và Torch/Torchvision trong requirements. Chỉ dùng phiên bản Python này để tránh lệch MediaPipe/joblib. Windows có lệnh cài dưới đây nhưng chưa được kiểm thử thực tế trong nhóm; cần môi trường desktop để mở cửa sổ OpenCV.

```bash
git clone https://github.com/Thanhmay2406/Final_computer_vision.git
cd Final_computer_vision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/local.txt
python -m pip install -r requirements/local-torch.txt
```

Trên Windows PowerShell, sau khi clone/cd:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/local.txt
.\.venv\Scripts\python.exe -m pip install -r requirements/local-torch.txt
```

Trong các lệnh tiếp theo, dùng `python` của môi trường vừa tạo. Với PowerShell, thay `python` bằng `.\.venv\Scripts\python.exe` nếu chưa kích hoạt venv. Tránh cài thêm `opencv-python-headless`: demo cần cửa sổ hiển thị. File `requirements/tools.txt` chỉ dành cho Kaggle API, không cần để mở webcam.

## 2. Tải checkpoint full một lần

Tải [handgesture-webcam-manual-002.zip](https://github.com/Thanhmay2406/Final_computer_vision/releases/download/manual-002/handgesture-webcam-manual-002.zip) từ [Release manual-002](https://github.com/Thanhmay2406/Final_computer_vision/releases/tag/manual-002), đặt ZIP ở gốc repo. Gói chứa ba model RF/MLP/CNN, scaler MLP và hồ sơ kết quả cần để kiểm chứng. CNN là MobileNetV3-Small đã fine-tune; **không cần tải weights ImageNet để chạy webcam**. Detector đã có ở `assets/mediapipe/hand_landmarker.task`.

Nếu dùng GitHub CLI, có thể tải bằng:

```bash
gh release download manual-002 --repo Thanhmay2406/Final_computer_vision --pattern handgesture-webcam-manual-002.zip
```

Kiểm tra SHA-256 theo [manifest](artifacts/manual_002.json), rồi giải nén:

```bash
python -c "import hashlib,json,pathlib; m=json.loads(pathlib.Path('artifacts/manual_002.json').read_text()); p=pathlib.Path(m['asset']); assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'], 'SHA-256 mismatch'; print('SHA-256 OK')"
python -m zipfile -e handgesture-webcam-manual-002.zip .
python scripts/train_v2.py verify --results-root models/manual_002
```

Lệnh verify kiểm tra checkpoint/config/mapping, CSV dự đoán, metric và confusion matrix. Không truyền dataset thì không kiểm chứng ảnh/ID với dataset nguồn. Cấu trúc sau giải nén:

```text
models/manual_002/
  rf/       # model.joblib + config/metrics
  mlp/      # model.pt + scaler.joblib + config/metrics
  cnn/      # MobileNet checkpoint + config/metrics
  comparison/
```

## 3. Chạy webcam

```bash
python scripts/webcam_demo.py --model mlp --mode count --checkpoint-dir models/manual_002 --camera 0 --width 640 --height 480
```

| Tùy chọn/thao tác | Ý nghĩa |
|---|---|
| `--model rf`, `--model mlp`, `--model cnn` | Chạy một model mỗi lượt để so sánh |
| `--mode count` | Chỉ hiện số **0–5**, mặc định |
| `--mode rps` | Chỉ hiện **Búa/Kéo/Bao**; tư thế 1/3/4 hiện “Ngoài bộ cử chỉ” |
| Phím `m` | Đổi count/rps trong lúc chạy |
| Phím `q` hoặc Esc | Đóng cửa sổ và giải phóng camera |
| `--camera 1` | Thử camera khác nếu camera0 không đúng thiết bị |

Quy ước lớp: `fist=0/búa`, `one=1`, `peace=2/kéo`, `three=3`, `four=4`, `palm=5/bao`. Đây là hai cách hiển thị của cùng model sáu lớp, chưa có logic trò chơi.

Confidence mặc định **0,7**; cửa sổ **7 frame**, cần **5 phiếu** cùng nhãn. Không có tay hiện `no_hand`; chưa đủ điều kiện hiện `uncertain`. Chỉ lật hình hiển thị, không lật đầu vào model. Ví dụ chọn MLP để chạy thử, chưa phải model triển khai được chốt.

Công cụ **không lưu ảnh/video**. Mặc định lưu `frames.csv`, `summary.json`, `protocol.json` tại `results_v2/webcam/<thời-điểm>/`. CSV giữ `label` gốc và thêm `display_mode`, `display_label`; summary ghi chế độ đầu/cuối và số lần đổi. Nếu lỗi mở camera, đóng ứng dụng khác đang dùng camera và kiểm tra quyền truy cập thiết bị.

## 4. Thực nghiệm webcam có lưu số liệu

Mỗi lần chạy chọn output-dir mới; không ghi đè lượt trước. Ví dụ 30 frame warm-up và 600 frame đo:

```bash
python scripts/webcam_demo.py --model mlp --mode count --checkpoint-dir models/manual_002 --camera 0 --width 640 --height 480 --warmup 30 --frames 600 --output-dir results_v2/team/mlp-01
```

Chạy lại với RF/CNN và output-dir khác. Làm sáu cử chỉ, mất tay, đưa tay trở lại, đổi mode và thử q/Esc trong một lượt không giới hạn frame. Ghi rõ cử chỉ đúng/sai/không ổn định, ánh sáng/nền/người thực hiện. Xem [checklist nghiệm thu](docs/acceptance/webcam_protocol.md).

Các lượt live khác động tác chưa phải so sánh cùng đầu vào. Không dùng tỷ lệ `recognized` hoặc số nhãn xuất hiện để tính Accuracy webcam. Cần ground truth và protocol riêng nếu muốn báo Accuracy thực tế.

## 5. Số liệu để viết báo cáo

| Model final | Test Accuracy | Test macro-F1 | CPU classifier median / p95 |
|---|---:|---:|---:|
| RF | 91,63% | 0,8975 | 7,593 / 7,955 ms |
| MLP | 92,83% | 0,9107 | 0,092 / 0,105 ms |
| CNN MobileNetV3-Small | 95,57% | 0,9461 | 3,366 / 3,923 ms |

Nguồn: [comparison CSV](reports/manual_002/final/comparison/comparison.csv), [latency CSV](reports/manual_002/benchmarks/cpu/classifier/classifier_latency.csv), [protocol CPU](reports/manual_002/benchmarks/cpu/classifier/classifier_protocol.json). CPU benchmark: i7-1255U, một thread, batch1, 20 warm-up, cùng 200 mẫu. MLP có scaler; chuẩn bị tensor CNN không thuộc classifier timing.

Các file nên dùng:

- [Báo cáo tổng hợp](docs/reports/full_manual_002.md): phương pháp, chất lượng, CPU, webcam và giới hạn.
- [Metric/history/dự đoán từng model](reports/manual_002/final/): đủ để lập bảng/biểu đồ và tính lại metric.
- [Ba CNN trên validation](reports/manual_002/cnn_candidates/): giải thích vì sao chọn MobileNet; không dùng test để chọn lại.
- [Thống kê dataset](reports/manual_002/dataset/preparation_report.json), [detector theo lớp](reports/manual_002/acceptance/detector_by_class.csv).
- [Ba lượt webcam thực tế](reports/manual_002/acceptance/report.md): 640×480,600 frame/model, khoảng16,65FPS; đầu vào khác nhau, overlay cũ.
- [Đối chiếu đề cương](docs/proposal_audit.md): phân biệt phần đạt, hạn chế và mục còn chờ.

**Giới hạn phải ghi trong báo cáo:** Accuracy95,57% tính trên3.680 ảnh test đã qua detector. Nguồn38.687 ảnh, giữ24.755, loại13.932 `no_hand`; không có subject/session metadata. Nếu coi detector từ chối là thất bại, CNN đúng3.517/5.804 ảnh test nguồn≈60,60%. Đây là chẩn đoán offline theo nhãn nguồn, không phải Webcam Accuracy. Xem [source-test diagnostic](reports/manual_002/acceptance/source_test_diagnostic.csv).

Hiện chưa có benchmark cùng video, xác nhận đầy đủ sáu cử chỉ/phím thoát hoặc deployment cuối. MLP/CNN đạt mục tiêu Accuracy92% trên tập được giữ; RF chưa đạt. Không suy FPS từ nghịch đảo latency classifier, không so mức cải thiện trực tiếp với baseline khác preprocessing/tập đánh giá.

## 6. Dataset, benchmark và train lại

Dataset ảnh/landmark lớn **không có trong Git hoặc gói webcam**. Người chỉ chạy webcam/viết báo cáo có thể dùng ngay các phần trên. Để chạy lại benchmark200 ảnh hoặc train, cần tự gắn/lấy dataset Kaggle đúng phiên; fingerprint giống nhau không thay thế manifest hash.

Notebook [07_interactive_workflow.ipynb](notebooks/kaggle_v2/07_interactive_workflow.ipynb) chạy độc lập trên Kaggle. Gắn hai Input `thanhmay2406/hand-gesture` và `thanhmay2406/hand-gesture-v2-runtime-assets-20261015` nếu tài khoản có quyền truy cập. Notebook mặc định smoke; full dùng `SMOKE_TEST=False`, RUN_ID mới, test tắt cho tới khi chốt validation. Giữ NUM_WORKERS0, GPU và môi trường Python3.11/3.12. Hướng dẫn kernel và protocol tại [pipeline v2](docs/pipeline_v2.md).

Khi đã có dataset dẫn xuất khớp model ở `data_v2/`, chạy benchmark vào thư mục mới:

```bash
python scripts/benchmark_v2.py --results-root models/manual_002 --data-root data_v2 --output-dir results_v2/team/benchmark-01
```

Thêm `--video /path/to/hand-video.mp4 --display` để đo cùng video cho ba model. Cần ít nhất330 frame:30 warm-up+300 đo. Chỉ chốt deployment sau benchmark đủ frame/có tay và đạt15FPS theo protocol. Công cụ không tự tạo/ghi video webcam.

## 7. Source, kiểm thử và phạm vi public

```text
handgesture/           # preprocessing, data, models, train, webcam, benchmark
scripts/               # CLI, notebook generator và Kaggle tools
notebooks/kaggle_v2/    # notebook phân phối, output trống
requirements/          # dependency local/Kaggle/tools
docs/                  # protocol, phân tích và checklist
reports/manual_002/    # số liệu/plot gọn để nhóm viết báo cáo
artifacts/manual_002.json  # SHA-256 và thông tin gói checkpoint
assets/mediapipe/      # detector dùng cho webcam
```

```bash
python -m pytest -q
```

Test không cần camera, GPU, mạng hoặc dataset tải ngoài. Để sinh lại notebook: `python scripts/build_kaggle_notebooks.py`, `python scripts/build_v2_notebooks.py`. Notebook đã thực thi chứa ảnh nguồn, gallery ảnh, baseline local, raw dataset, video và đề cương có thông tin sinh viên không được đưa lên phần public này. Checkpoint ở Release, các lượt thực nghiệm mới và venv được gitignore. [Bản đồ source](docs/source_map.md), [CLI](scripts/README.md), [notebook](notebooks/README.md).
