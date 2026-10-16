# Kết quả MLP

## Cấu hình

Phiên: `mlp`; seed: 42.
Kiến trúc: mlp.
Preprocessing: {'detector_sha256': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1', 'version': 'pixel_wrist_max3d_square_v2', 'detection_confidence': 0.5, 'presence_confidence': 0.5, 'tracking_confidence': 0.5, 'roi_scale': 1.4, 'epsilon': 1e-06, 'num_hands': 1, 'mirror_input': False, 'detector_input_padding': 0.5, 'detector_runtime_version': '0.10.21'}.
Optimizer: AdamW; learning rate ban đầu: 0.001; weight decay: 0.0001; batch size: 256.
Epoch tối đa: 60; patience: 8; checkpoint được chọn tại epoch: 31.
Scheduler: `null`.

[Cấu hình đầy đủ](model_config.json)

## Kết quả

| model | split | sample_count | accuracy | macro_precision | macro_recall | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | validation | 3690 | 0.938753 | 0.924049 | 0.924295 | 0.923535 |
| mlp | test | 3680 | 0.928261 | 0.911715 | 0.911321 | 0.910729 |

[Metric từng lớp](per_class_metrics.csv) · [Metric gốc](metrics.json) · [Dự đoán test](test_predictions.csv)

## Phân tích nhầm lẫn

| model | true | predicted | count |
| --- | --- | --- | --- |
| mlp | one | fist | 38 |
| mlp | peace | one | 32 |
| mlp | three | peace | 29 |

## Thời gian và thiết bị

Training: 12.590 s; fit wall time: 14.714 s.
Device: cuda; CPU: Intel(R) Xeon(R) CPU @ 2.00GHz; GPU: Tesla T4.
RF đo estimator.fit; mạng đo train batches gồm loader, không gồm validation. Fit wall time của mạng gồm validation và chọn checkpoint. Không so sánh tốc độ train công bằng khi phần cứng khác nhau.

## Biểu đồ

[Lịch sử huấn luyện](history.png)
[Confusion matrix](confusion_matrix.png)

[Dữ liệu lịch sử](history.csv)

RF chỉ có một lượt fit; epoch/loss/learning rate không áp dụng. Train metrics của MLP/CNN được đo trong batch có dropout/augmentation; validation/test ở eval. Ô trống là số đo không áp dụng hoặc chưa có, không phải 0. Test được đánh giá sau khi model/checkpoint cố định. Split chưa tách người chụp; kết quả không đại diện FPS hoặc webcam end-to-end.

## Các cặp nhầm lẫn nhiều nhất
