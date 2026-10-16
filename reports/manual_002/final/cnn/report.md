# Kết quả CNN

## Cấu hình

Phiên: `mobilenet_v3_small`; seed: 42.
Kiến trúc: mobilenet_v3_small.
Preprocessing: {'detector_sha256': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1', 'version': 'pixel_wrist_max3d_square_v2', 'detection_confidence': 0.5, 'presence_confidence': 0.5, 'tracking_confidence': 0.5, 'roi_scale': 1.4, 'epsilon': 1e-06, 'num_hands': 1, 'mirror_input': False, 'detector_input_padding': 0.5, 'detector_runtime_version': '0.10.21'}.
Optimizer: AdamW; learning rate ban đầu: 0.001; weight decay: 0.0001; batch size: 64.
Epoch tối đa: 40; patience: 8; checkpoint được chọn tại epoch: 38.
Scheduler: `null`.

[Cấu hình đầy đủ](model_config.json)

## Kết quả

| model | split | sample_count | accuracy | macro_precision | macro_recall | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| cnn | validation | 3690 | 0.954743 | 0.945439 | 0.946459 | 0.945704 |
| cnn | test | 3680 | 0.955707 | 0.946436 | 0.945908 | 0.946105 |

[Metric từng lớp](per_class_metrics.csv) · [Metric gốc](metrics.json) · [Dự đoán test](test_predictions.csv)

## Phân tích nhầm lẫn

| model | true | predicted | count |
| --- | --- | --- | --- |
| cnn | one | fist | 19 |
| cnn | fist | one | 16 |
| cnn | three | fist | 12 |

## Thời gian và thiết bị

Training: 4830.851 s; fit wall time: 5395.272 s.
Device: cuda; CPU: Intel(R) Xeon(R) CPU @ 2.00GHz; GPU: Tesla T4.
RF đo estimator.fit; mạng đo train batches gồm loader, không gồm validation. Fit wall time của mạng gồm validation và chọn checkpoint. Không so sánh tốc độ train công bằng khi phần cứng khác nhau.

## Biểu đồ

[Lịch sử huấn luyện](history.png)
[Confusion matrix](confusion_matrix.png)

[Dữ liệu lịch sử](history.csv)

RF chỉ có một lượt fit; epoch/loss/learning rate không áp dụng. Train metrics của MLP/CNN được đo trong batch có dropout/augmentation; validation/test ở eval. Ô trống là số đo không áp dụng hoặc chưa có, không phải 0. Test được đánh giá sau khi model/checkpoint cố định. Split chưa tách người chụp; kết quả không đại diện FPS hoặc webcam end-to-end.

## Các cặp nhầm lẫn nhiều nhất
