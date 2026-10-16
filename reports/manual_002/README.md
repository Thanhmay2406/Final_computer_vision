# Số liệu full manual_002 dùng viết báo cáo

Đây là bản sao số liệu đã đo, không phải thư mục checkpoint. Model tải ở GitHub Release manual-002; ảnh nguồn/landmark manifest đầy đủ không nằm trong Git.

- final/: config, metric, history, validation/test prediction, confusion matrix và report cho RF/MLP/CNN được chọn.
- cnn_candidates/: số liệu validation và config ba CNN; không có weights. Không gọi read_run lên bản sao report này.
- benchmarks/: CPU classifier và comparison đã đo.
- acceptance/: ba lượt live600 frame và chẩn đoán detector; xác nhận đúng/sai từng cử chỉ còn chờ.
- dataset/preparation_report.json: thống kê, split, fingerprint và hash metadata của dữ liệu Kaggle.

Ảnh gallery/cặp nhầm và notebook đã chạy chứa ảnh nguồn được giữ ở bản local, không nằm trong gói public. Các plot thống kê vẫn có. Các liên kết tới gallery được bỏ khỏi bản sao Markdown này; JSON/CSV và measured history không chỉnh sửa. Đường dẫn Kaggle và hash checkpoint trong record là provenance gốc, không phải đường dẫn model tại máy bạn.

Classifier Accuracy CNN95,57% tính trên3.680 ảnh test qua detector. Không suy ra Webcam Accuracy. Benchmark same-video/deployment chưa chốt; ba lượt webcam riêng không cùng đầu vào. Các lượt live_01 dùng overlay trước khi tách hai chế độ count/rps.
