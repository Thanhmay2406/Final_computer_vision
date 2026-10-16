# Kết quả RF

## Cấu hình

Phiên: `rf`; seed: 42.
Kiến trúc: rf.
Preprocessing: {'detector_sha256': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1', 'version': 'pixel_wrist_max3d_square_v2', 'detection_confidence': 0.5, 'presence_confidence': 0.5, 'tracking_confidence': 0.5, 'roi_scale': 1.4, 'epsilon': 1e-06, 'num_hands': 1, 'mirror_input': False, 'detector_input_padding': 0.5, 'detector_runtime_version': '0.10.21'}.
Số cây: 300; max depth: 20.

[Cấu hình đầy đủ](model_config.json)

## Kết quả

| model | split | sample_count | accuracy | macro_precision | macro_recall | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| rf | validation | 3690 | 0.918970 | 0.897673 | 0.900626 | 0.898817 |
| rf | test | 3680 | 0.916304 | 0.896919 | 0.898286 | 0.897465 |

[Metric từng lớp](per_class_metrics.csv) · [Metric gốc](metrics.json) · [Dự đoán test](test_predictions.csv)

## Phân tích nhầm lẫn

| model | true | predicted | count |
| --- | --- | --- | --- |
| rf | peace | one | 35 |
| rf | three | four | 33 |
| rf | peace | three | 30 |

## Thời gian và thiết bị

Training: 15.680 s; fit wall time: 16.168 s.
Device: cpu; CPU: Intel(R) Xeon(R) CPU @ 2.00GHz; GPU: N/A.
RF đo estimator.fit; mạng đo train batches gồm loader, không gồm validation. Fit wall time của mạng gồm validation và chọn checkpoint. Không so sánh tốc độ train công bằng khi phần cứng khác nhau.

## Biểu đồ

[Lịch sử huấn luyện](history.png)
[Confusion matrix](confusion_matrix.png)

[Dữ liệu lịch sử](history.csv)

RF chỉ có một lượt fit; epoch/loss/learning rate không áp dụng. Train metrics của MLP/CNN được đo trong batch có dropout/augmentation; validation/test ở eval. Ô trống là số đo không áp dụng hoặc chưa có, không phải 0. Test được đánh giá sau khi model/checkpoint cố định. Split chưa tách người chụp; kết quả không đại diện FPS hoặc webcam end-to-end.

## Các cặp nhầm lẫn nhiều nhất
