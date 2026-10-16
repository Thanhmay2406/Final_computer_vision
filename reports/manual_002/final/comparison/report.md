# So sánh RF, MLP và CNN

Cùng manifest, seed, mapping và tập test. CNN hiện tại là phiên `mobilenet_v3_small`.

## Chất lượng

| model | validation_accuracy | validation_macro_f1 | accuracy | macro_f1 | test_samples |
| --- | --- | --- | --- | --- | --- |
| rf | 0.918970 | 0.898817 | 0.916304 | 0.897465 | 3680 |
| mlp | 0.938753 | 0.923535 | 0.928261 | 0.910729 | 3680 |
| cnn | 0.954743 | 0.945704 | 0.955707 | 0.946105 | 3680 |

[Bảng tổng hợp đầy đủ](comparison.csv) · [Metric từng lớp](per_class_metrics.csv)

## Thời gian và thiết bị

| model | training_time_s | fit_wall_time_s | training_device | training_cpu | training_gpu |
| --- | --- | --- | --- | --- | --- |
| rf | 15.679556 | 16.167725 | cpu | Intel(R) Xeon(R) CPU @ 2.00GHz | N/A |
| mlp | 12.590460 | 14.713793 | cuda | Intel(R) Xeon(R) CPU @ 2.00GHz | Tesla T4 |
| cnn | 4830.850767 | 5395.272080 | cuda | Intel(R) Xeon(R) CPU @ 2.00GHz | Tesla T4 |

Thời gian train mô tả từng phiên; phần cứng khác nhau nên không dùng làm kết luận tốc độ công bằng. RF/MLP dùng landmark, CNN dùng ảnh. Không tính detector, crop hoặc pipeline webcam.

## Phân tích lỗi

| model | true | predicted | count |
| --- | --- | --- | --- |
| rf | peace | one | 35 |
| rf | three | four | 33 |
| rf | peace | three | 30 |
| mlp | one | fist | 38 |
| mlp | peace | one | 32 |
| mlp | three | peace | 29 |
| cnn | one | fist | 19 |
| cnn | fist | one | 16 |
| cnn | three | fist | 12 |

[Biểu đồ chất lượng](comparison.png) · [Confusion matrix cùng thang màu](confusion_matrices.png)

## Báo cáo từng model

[RF](../rf/report.md)
[MLP](../mlp/report.md)
[CNN](../cnn/report.md)

Chưa chạy benchmark latency trong báo cáo này; không suy ra latency hoặc FPS.
