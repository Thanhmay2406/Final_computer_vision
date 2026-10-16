# Huấn luyện và so sánh trên Kaggle

Bốn notebook trong `notebooks/kaggle/` chạy độc lập: code cần thiết đã được nhúng, không cần tải thư mục `scripts/` lên Kaggle.

| Notebook | Đầu vào mô hình | Thiết bị nên chọn | Output |
|---|---|---|---|
| `01_random_forest.ipynb` | 63 tọa độ hiện có | CPU | `hand_gesture/rf/` |
| `02_mlp.ipynb` | 63 tọa độ, StandardScaler fit train | GPU nếu có | `hand_gesture/mlp/` |
| `03_cnn.ipynb` | RGB 128×128 | GPU | `hand_gesture/cnn/` |
| `04_compare_models.ipynb` | Dataset và output ba notebook | CPU | `hand_gesture/comparison/` |

## Chạy ba notebook huấn luyện

1. Tạo notebook Kaggle, dùng **File → Import Notebook** để nhập từng file `.ipynb`.
2. Trong **Add Input**, gắn [dataset hand-gesture](https://www.kaggle.com/datasets/thanhmay2406/hand-gesture). Cần có `metadata/manifest.csv`, `metadata/preparation_report.json` và `images/` trong cùng thư mục gốc.
3. RF chạy CPU; MLP/CNN chọn GPU trong Settings → Accelerator. `DEVICE='auto'` tự chọn CUDA nếu khả dụng, còn lại dùng CPU. [Tài liệu Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) mô tả môi trường notebook và accelerator.
4. Kiểm tra cell cấu hình đầu tiên. Để `SMOKE_TEST=False` khi chạy thực nghiệm đầy đủ; giữ seed 42 và baseline đã chọn. Không cần Internet, pretrained weights hoặc tải mã dự án.
5. Chạy toàn bộ cell. Nếu tìm thấy nhiều dataset phù hợp, đặt `DATA_ROOT` bằng đường dẫn thư mục chứa **cả `metadata/` và `images/`**, rồi chạy lại toàn bộ.
6. Dùng **Save Version → Save & Run All** để lưu output. Artifact nằm dưới `/kaggle/working/hand_gesture/<model>/`.

RF: 300 cây, max depth 20, min samples leaf 2, max features sqrt. MLP: 63→128→64→6, dropout 0,2, batch 256, tối đa 60 epoch; AdamW lr=1e-3, early stopping patience 8, không scheduler. CNN: ba block 32/64/128 kênh, global average pooling, dropout 0,2, batch 64, tối đa 50 epoch; AdamW lr=3e-4, early stopping patience 12. Hai mạng giữ weight decay=1e-4 và CrossEntropyLoss; chọn macro-F1 validation cao nhất, validation loss phá hòa. Không áp dụng class weight hoặc oversampling.

CNN dùng ReduceLROnPlateau theo validation macro-F1: `mode='max'`, `factor=0.5`, `patience=3`, `threshold=1e-4`, `threshold_mode='abs'`, `min_lr=1e-6`. Scheduler cập nhật sau validation, learning rate mới có hiệu lực ở epoch kế tiếp. `history.csv` có cột `learning_rate` ghi giá trị thực dùng trong epoch; `history.png` bổ sung đồ thị learning rate. `model_config.json` lưu cấu hình optimizer, scheduler, early stopping và epoch được chọn. MLP giữ nguyên mặc định huấn luyện.

CNN có `RUN_NAME='cnn_tuned'` trong metadata; thư mục output chuẩn là `/kaggle/working/hand_gesture/cnn/`, cùng cách bố trí với RF/MLP. Cấu hình tuning không đổi. Phiên ngày 15/10/2026: validation macro-F1 **0,738110**, test accuracy **72,864%**, test macro-F1 **0,727011**, chọn epoch 40/50 trên Tesla T4. Một lượt chạy chưa tách riêng tác động learning rate và scheduler.

Phiên riêng tư đã chạy tại [Hand Gesture CNN Tuned 20261015](https://www.kaggle.com/code/thanhmay2406/hand-gesture-cnn-tuned-20261015) xuất theo cấu trúc cũ `hand_gesture/cnn_tuned/`. Artifact này đã được chuẩn hóa về `results/cnn/`; bản lưu nguyên trạng nằm tại `results/archive/cnn_tuned_20261015/`, gồm báo cáo so với baseline cũ, log và trạng thái phiên. Workspace hiện không có checkpoint CNN baseline cũ; số đo tham chiếu trong báo cáo lưu trữ không thay thế checkpoint đó.

Split giữ nguyên **train 27.080 / validation 5.803 / test 5.804**. Notebook kiểm tra SHA-256 manifest so với báo cáo, schema, landmark hữu hạn, mapping, ID/hash/path không trùng và ảnh tồn tại, SHA-256 toàn bộ ảnh khớp manifest. Không tự loại mẫu hoặc chia lại. Hash ảnh bảo đảm giữ nguyên nội dung đã được pipeline chuẩn bị giải mã trước khi upload; notebook không giải mã lại toàn bộ ảnh RF/MLP.

MLP fit scaler chỉ trên train. CNN áp dụng EXIF orientation, RGB, resize bilinear 128×128, pixel/255; chỉ train được xoay ±10° và điều chỉnh sáng/tương phản ±15%, không flip. Train metrics của mạng được thu trong các batch có dropout/augmentation; validation/test ở chế độ eval. Test chỉ được đánh giá sau khi checkpoint đã cố định, không refit train+validation.

## Artifact và tải model

Mỗi model xuất `model_config.json`, `class_mapping.json`, `metrics.json`, `test_predictions.csv`, `history.csv`, `history.png`, confusion matrix số lượng/tỷ lệ, `summary.csv`, `per_class_metrics.csv` và `report.md`. Gallery tối đa 12 lỗi xuất khi có dự đoán sai. Checkpoint và scaler theo từng loại model được giữ nguyên.

- `summary.csv`: hai dòng validation/test, cùng accuracy, macro-precision/recall/F1 và số mẫu.
- `per_class_metrics.csv`: validation rồi test, mỗi split theo thứ tự `fist/one/peace/three/four/palm`, gồm precision, recall, F1 và support.
- `history.csv`: cùng các cột `step`, `epoch`, `learning_rate`, `train_loss`, `validation_loss`, `train_accuracy`, `validation_accuracy`, `train_macro_f1`, `validation_macro_f1`, `smoke_test`. RF có một bước fit; epoch/loss/learning rate để trống. MLP cũ được bổ sung LR từ cấu hình cố định; khi scheduler tồn tại nhưng không ghi LR thì để trống, không suy đoán.
- `report.md`: cùng mẫu cấu hình, kết quả, nhầm lẫn, thời gian/thiết bị và liên kết biểu đồ. MLP/CNN có bốn panel loss, accuracy, macro-F1, learning rate; RF có train/validation tại một fit và chú thích không áp dụng. Canvas, font và màu được thống nhất theo loại biểu đồ.

Các số đo gốc trong `metrics.json` và CSV dự đoán không bị ghi lại khi chuẩn hóa. Báo cáo kiểm tra metric test khớp dự đoán trước khi tạo output dẫn xuất. Gallery lỗi không được tạo nếu không có dự đoán sai; báo cáo chỉ liên kết các ảnh có thật.

- RF: `joblib.load('model.joblib')`; đưa vào 63 cột `lm_0`–`lm_62` đúng thứ tự, giữ nguyên tọa độ.
- MLP: tải `scaler.joblib`, transform 63 cột rồi đổi float32; khởi tạo `GestureMLP`, tải `model.pt`, gọi `eval()`.
- CNN: khởi tạo `GestureCNN`, tải `model.pt`, gọi `eval()`; dùng `image_tensor(image, training=False)` với cùng preprocessing.

Các lớp và hàm suy luận nằm sẵn trong notebook. `model.pt` chỉ chứa `state_dict`; tải bằng `torch.load(..., map_location='cpu', weights_only=True)`. Đây là [cách lưu/tải model PyTorch được khuyến nghị](https://docs.pytorch.org/tutorials/beginner/saving_loading_models). Chỉ tải artifact joblib từ nguồn tin cậy. Kiến trúc được ghi trong notebook và `model_config.json`; muốn thay kiến trúc phải cập nhật code tải tương ứng.

## Notebook tổng hợp — local và Kaggle

**Local:** mở `notebooks/kaggle/04_compare_models.ipynb` bằng kernel Python có numpy, pandas, scikit-learn, Pillow, joblib, matplotlib, IPython rồi chọn **Run All**. Kernel có thể bắt đầu tại gốc dự án hoặc `notebooks/kaggle/`. Notebook tìm thư mục `results/` từ cwd lên các thư mục cha, đọc trực tiếp `rf/`, `mlp/`, `cnn/` và ghi báo cáo vào `results/comparison/`. Mặc định `RUN_BENCHMARK=False`: không đọc dataset, không import Torch, không giải tuần tự checkpoint/joblib và không huấn luyện lại model. Model/scaler vẫn phải có mặt để kiểm tra SHA-256.

Có thể chỉ định đường dẫn ở cell cấu hình đầu:

```python
RESULTS_ROOT = '/duong/dan/toi/results'  # None: tự tìm local rồi mới tìm Kaggle
RESULT_DIRS = None  # Khi chỉ định dict rf/mlp/cnn, dict này được ưu tiên cao nhất
OUTPUT_ROOT = None  # None: dùng results/ local; notebook tạo comparison/ bên trong
RUN_BENCHMARK = False
DATA_ROOT = None  # Chỉ dùng khi bật benchmark
```

Thứ tự chọn input: `RESULT_DIRS` → `RESULTS_ROOT` → `results/` local → `/kaggle/input`. Local không quét archive để thay model bị thiếu; notebook báo đường dẫn artifact bị thiếu. Output chỉ định bằng `OUTPUT_ROOT` luôn được ưu tiên. Nếu chỉ có `RESULT_DIRS` thủ công mà không có root local hoặc Kaggle, output mặc định là `./results/comparison/`. Notebook in input/output và chế độ benchmark trước khi xử lý.

**Kaggle:** nhập notebook 04, gắn **Notebook Output** từ ba phiên bằng Add Input. Chế độ mặc định không cần gắn dataset gốc. Notebook tự tìm input trong `/kaggle/input` (bỏ qua `archive/`), output ghi vào `/kaggle/working/hand_gesture/comparison/`. Nếu gắn nhiều phiên cùng model, chọn rõ:

```python
RESULT_DIRS = {
    'rf': '/kaggle/input/<rf-output>/hand_gesture/rf',
    'mlp': '/kaggle/input/<mlp-output>/hand_gesture/mlp',
    'cnn': '/kaggle/input/<cnn-output>/hand_gesture/cnn',
}
```

Mọi chế độ đều kiểm tra hash model/scaler/config/mapping/predictions, cùng seed, mapping, manifest hash, smoke/full, ID và nhãn test; các metric test phải khớp CSV dự đoán. Mặc định chỉ xác minh tính nhất quán giữa kết quả đã lưu, chưa đối chiếu dataset gốc. Output gồm `comparison.csv`, `per_class_metrics.csv`, `top_confusions.csv`, `report.md`, biểu đồ chất lượng và confusion matrix chung. Notebook không sửa artifact nguồn.

**Benchmark tùy chọn:** đặt `RUN_BENCHMARK=True` và chạy lại toàn bộ notebook bằng kernel có Torch. Dataset được tìm tại `data/` local hoặc input Kaggle; có thể chỉ định `DATA_ROOT` tới thư mục chứa `metadata/` và `images/`. Chỉ bật với checkpoint/joblib tin cậy. Notebook kiểm tra dataset, đối chiếu manifest và toàn bộ nhãn/ID test trước khi tải model để đo. Thiếu dataset, sai protocol hoặc checkpoint không tương thích gây lỗi rõ ràng, không tự tắt benchmark.

Benchmark đo cùng CPU, một thread, batch 1, warm-up 20 lượt và 200 mẫu test cố định theo seed 42; smoke dùng số mẫu test nhỏ hiện có. MLP tính cả scaler; CNN dùng tensor đã chuẩn bị. Không tính đọc ảnh/crop/MediaPipe. Latency và `benchmark_protocol.json` chỉ xuất khi thực sự benchmark. Khi benchmark tắt, bảng/báo cáo không chứa latency cũ và file `benchmark_protocol.json` cũ trong output comparison được gỡ. Đây không phải FPS toàn hệ thống.

Thời gian train chỉ mô tả phiên đã lưu: RF đo estimator.fit; mạng đo train batches gồm loader và không gồm validation, còn fit wall time gồm validation/checkpoint selection. Không suy ra tốc độ train công bằng giữa phần cứng khác nhau.

## Chuẩn hóa kết quả cục bộ

Bắt đầu xem tại `results/README.md` hoặc `results/comparison/report.md`. Thư mục chuẩn:

```text
results/
  rf/           # artifact và báo cáo RF
  mlp/          # artifact và báo cáo MLP
  cnn/          # artifact và báo cáo CNN tinh chỉnh
  comparison/   # bảng và biểu đồ so sánh ba model
  archive/      # bản lưu nguyên trạng, không tham gia tự tìm output
```

Tạo lại output đánh giá từ artifact hiện có, không huấn luyện hay benchmark lại:

```bash
~/.venv/bin/python scripts/normalize_results.py
```

Có thể truyền `--results-root` và `--data-root`. Script kiểm tra hash, protocol, cấu hình, metric test và ảnh gallery trước khi ghi. Nếu CNN còn lồng thư mục, script sao lưu và xác nhận hash toàn bộ bản lưu trước khi chuyển artifact ra `results/cnn/`. Bản lưu khác nội dung hoặc xung đột tên artifact khiến script dừng. Chạy lại không tạo bản lưu trùng hoặc thêm thư mục lồng.

`comparison.csv` cục bộ gồm metric validation/test, số mẫu test, kích thước artifact, thời gian và thiết bị train. Script không thêm latency/FPS chưa đo. Notebook 04 vẫn có thể đo classifier latency và thêm kết quả benchmark thực tế vào bảng/báo cáo.

## Kiểm chứng cục bộ và giới hạn

Không chạy baseline đầy đủ trên CPU cục bộ trong bước tạo notebook. Để chạy lại kiểm thử:

```bash
~/.venv/bin/python -m pytest -q
~/.venv/bin/python scripts/smoke_test_kaggle_notebooks.py
```

Smoke test dùng 8/4/4 mẫu **mỗi lớp**, lấy trong train/validation/test hiện có, tổng 96 mẫu; RF 10 cây, MLP/CNN 2 epoch, CPU. Cả bốn notebook chạy bằng kernel `~/.venv`, output lưu trong thư mục tạm và có `smoke_test: true`. Script kiểm tra dự đoán sau tải lại model trùng output và scaler fit đúng train; notebook 04 được bật benchmark để kiểm thử nhánh dataset/model. Bộ kiểm thử riêng xác nhận chế độ mặc định chạy được khi không có dataset và bị chặn import Torch/tải model. Notebook giao cho người dùng giữ output trống, `SMOKE_TEST=False`. Test CPU/GPU bao phủ logic chọn thiết bị bằng mô phỏng availability; cần chạy GPU thật trên Kaggle để xác nhận môi trường CUDA.

Split hiện tại chưa bảo đảm tách người chụp/phiên; cần protocol theo nhóm khi có metadata. Chưa xác minh công thức chuẩn hóa landmark/crop từ mã gốc. So sánh áp dụng cho các mẫu có cả ảnh và landmark, không đánh giá end-to-end detector. Không suy Accuracy từ log no-hand. Accuracy ≥92% chỉ được báo đạt/chưa đạt theo kết quả thật, không bảo đảm baseline đạt mục tiêu; output smoke không phải kết quả huấn luyện đầy đủ.
