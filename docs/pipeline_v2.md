# Thiết kế và quy trình chạy v2

## Dữ liệu và preprocessing

Nguồn là ảnh Kaggle hiện có trong `data/images/`; đây không phải ảnh webcam gốc. Preprocessing v2 chạy lại MediaPipe để xác định đầu vào có thể tái lập. Giữ nguyên `sample_id`, nhãn và split của mỗi mẫu nguồn. Không dùng lại landmark chưa rõ công thức từ CSV cũ.

Với ảnh W×H, landmark chuẩn hóa MediaPipe được chuyển thành `(xW,yH,zW)`. Trừ cổ tay, chia cho khoảng cách 3D lớn nhất tới cổ tay (epsilon 1e-6), flatten thành 63 float32. ROI lấy bbox x/y của 21 điểm, cạnh vuông bằng ceil(1,4 × cạnh bbox lớn nhất), căn giữa; phần ngoài ảnh padding đen. Crop lưu PNG để không thêm sai số JPEG giữa preprocessing offline/online. Không mirror trước suy luận, không đổi tay trái/phải. IMAGE offline và VIDEO online dùng cùng detector/config và cùng code hình học; tracking có thể cho landmark khác giữa hai chế độ.

Detector nhận ảnh có padding đen 50% mỗi chiều; landmark được ánh xạ ngược về tọa độ ảnh gốc trước khi tính đặc trưng/crop. Đây là điều chỉnh sau kiểm tra 72 ảnh nguồn: 14/72 phát hiện với ảnh crop trực tiếp, 49/72 khi padding50%. Áp dụng cùng quy tắc cho webcam và ghi padding/runtime MediaPipe vào fingerprint.

Fingerprint SHA-256 gồm version công thức, hash detector, các ngưỡng, ROI scale và epsilon. Loader kiểm tra fingerprint trước khi dùng webcam. ID/path/hash của mẫu phải duy nhất; crop giống byte sau xử lý bị loại toàn bộ để tránh giao nhau giữa các split. Metadata ghi số nguồn/chấp nhận/loại theo lý do và split. Nếu mất một lớp trong một split, không công bố dataset đích; cần xem log staging trước khi chọn protocol khác.

Các giới hạn vẫn còn: nguồn đã được crop, thiếu metadata người chụp, và phân bố webcam có thể khác ảnh nguồn. Không gọi random split là subject-wise hoặc dùng offline Accuracy thay cho webcam Accuracy.

## Source dùng chung

- `handgesture/preprocessing.py`: detector và hình học.
- `handgesture/data.py`: dataset dẫn xuất, journal, kiểm chứng hash và split.
- `handgesture/models.py`: registry, tensor ảnh, loader và inference CPU.
- `handgesture/training.py`: train/validation, chọn CNN, finalize test, kiểm chứng kết quả.
- `handgesture/live.py`: trạng thái dự đoán, overlay, camera lifecycle, timing từng frame.
- `handgesture/benchmark.py`: classifier/video benchmark và lựa chọn deployment.

Notebook builder nhúng nguyên source vào các module trong bộ nhớ; notebook không cần thư mục source tại Kaggle. Sửa source trước, sau đó chạy hai notebook builder. Không sửa bản nhúng bằng tay. Thư viện báo cáo cũ tiếp tục được dùng để giữ định dạng bảng/biểu đồ thống nhất.

## Train và lựa chọn

RF: 300 cây, depth20, leaf2, sqrt features. MLP: 63→128→64→6, dropout0,2, StandardScaler fit train, AdamW lr1e-3, 60 epoch/patience8. CNN GAP và spatial: 60 epoch/patience12, AdamW lr3e-4, weight decay1e-4 và ReduceLROnPlateau cấu hình như baseline. Spatial dùng AdaptiveAvgPool4×4→Linear128→ReLU→Dropout0,3→Linear6.

MobileNetV3-Small dùng đúng SHA-256 weights ImageNet ghi trong source. Resize ROI trực tiếp224×224, pixel/255, mean/std ImageNet, không center crop. Năm epoch đầu features đóng băng (gồm BN eval), classifier lr1e-3; sau đó fine-tune tối đa35 epoch, features lr1e-5, classifier lr1e-4, patience8. Chọn checkpoint tốt nhất xuyên suốt hai phase. Không download khi inference.

Augmentation train: xoay±10°, sáng/tương phản0,85–1,15, không flip. Seed42. Mỗi candidate chỉ ghi validation_predictions, không có test_predictions. So sánh ba CNN bằng validation macro-F1 rồi loss. File selection được ghi trước đánh giá test; finalize chỉ đánh giá RF/MLP/CNN thắng. Test đã xem ở baseline lịch sử, vì vậy không mô tả đây là một bộ test hoàn toàn chưa từng quan sát trong toàn bộ dự án.

## Cấu trúc output

```text
results_v2/<experiment>/
  runs/rf/                 # run.json, checkpoint, validation_predictions, history
  runs/mlp/
  candidates/gap/
  candidates/spatial/
  candidates/mobilenet_v3_small/
  final/
    selection.json
    rf/ mlp/ cnn/          # metrics, validation/test predictions, config, checkpoint, báo cáo
    comparison/
```

Ví dụ train local chỉ để smoke:

```bash
python scripts/train_v2.py train --model rf --data-root data_v2 --output-dir /tmp/run-rf --smoke
python scripts/train_v2.py train --model mlp --data-root data_v2 --output-dir /tmp/run-mlp --smoke --device cpu
python scripts/train_v2.py train --model cnn --architecture mobilenet_v3_small --weights assets/pretrained/mobilenet_v3_small-047dcff4.pth --data-root data_v2 --output-dir /tmp/run-mobile --smoke --device cpu
```

Finalization cần đủ ba CNN candidate cùng protocol. Đầu ra đã tồn tại không được ghi đè; dùng experiment mới. `train_v2.py verify` là read-only; `train_v2.py report` kiểm chứng artifact nguồn rồi tạo lại output dẫn xuất, giữ nguyên history và artifact đo. Baseline thiếu validation_predictions không được trộn vào pipeline v2. Các lượt smoke dùng weights fixture chỉ có ý nghĩa kiểm thử; trường pretrained ghi rõ TEST_FIXTURE_NOT_PRETRAINED và webcam từ chối checkpoint smoke.

## Đo và nghiệm thu

Classifier benchmark: CPU1 thread, batch1, warm-up20, 200 mẫu test cố định seed42 (hoặc số mẫu nhỏ của smoke). MLP tính scaler, CNN đã tạo tensor trước khi đo; không tính detector/IO. Video benchmark: cùng video,30 warm-up +300 frame, overlay được tính; display/waitKey chỉ tính khi bật display. FPS lấy số frame chia tổng thời gian các frame đo, không lấy trung bình nghịch đảo latency.

Chọn deployment trong model đạt15FPS, đủ300frame, ít nhất80% frame có tay hợp lệ và cả FPS tổng/FPS trên frame có tay đều đạt15, ưu tiên validation macro-F1 rồi median latency; smoke không được chọn. Video thiếu frame hoặc camera không mở được phải ghi lỗi/chưa hoàn tất. Webcam ghi timing, cấu hình threshold/window/votes, thiết bị và phiên bản; không tự lưu hình/video.

Kiểm tra thủ công trước bảo vệ: thử sáu lớp, hai tay riêng rẽ, nền/ánh sáng khác, đưa tay ra khỏi khung, confidence thấp, camera bị ngắt, phímq/Esc, đổi từng model. Ghi kết quả kiểm tra thực tế; test giả lập không thay thế phần này.

## Notebook chạy thủ công

Ưu tiên `notebooks/kaggle_v2/07_interactive_workflow.ipynb`: một notebook, từng cell native cho dependency, GPU, input, preprocessing, RF/MLP/ba CNN, validation, test có cờ xác nhận và export. Mặc định smoke, test tắt. Không có wrapper/script train nền; giữ protocol và shared source.

Sau preprocessing, notebook khóa context gồm thư mục phiên, chế độ smoke/full, seed, worker, device, đường dẫn cấu hình nguồn/asset và hash manifest. Đổi cấu hình phải chạy lại từ input/preprocessing; đổi riêng cờ đánh giá test được cho phép. Export lấy nhãn smoke từ protocol đã kiểm chứng. Notebook cần kernel Python3.11/3.12 và dependency đã khớp, không tự thay Python.

Phiên smoke tự động trước đây đã hủy; việc kiểm chứng notebook mới hiện dùng smoke local với detector/weights fixture, không đại diện nghiệm thu GPU Kaggle hay kết quả full.

### Kernel native cho image Kaggle Python3.13+

Kaggle interactive được kiểm tra ngày16/10/2026 dùng Python3.13.15. Hai cell tùy chọn của notebook07 cài Python3.11/dependency/CUDA Torch ở `/tmp` rồi dùng `execve` thay chính kernel, giữ file kết nối Jupyter. Cờ mặc định tắt; sau setup và chuyển cần chạy lại từ cấu hình. Việc chuyển đã được kiểm tra bằng một Jupyter kernel local thực và trên Kaggle: Python3.11.17, MediaPipe0.10.21, Torch2.11.0+cu128, TeslaT4, cùng PID50. Không chạy train qua script con.

Tìm Input dùng `os.walk` và bỏ qua cây `images` thay vì rglob toàn bộ ảnh. Trên Kaggle đã xác nhận bước cũ chậm ở `find_asset`; cách mới tìm đủ manifest/detector/weights. Tiền xử lý thật trên144 ảnh smoke giữ89, loại55 trong8.4s, đủ sáu lớp mỗi split. Chi tiết phiên native: `docs/implementation/kaggle_interactive_run.json`.
