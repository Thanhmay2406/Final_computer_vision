# Nghiệm thu webcam full manual_002

| model | measured_frames | hand_frames | hand_fraction | fps | median_ms | p95_ms | recognized | uncertain | no_hand | mean_capture_ms | mean_detector_ms | mean_preprocessing_ms | mean_classifier_ms | mean_render_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf | 600 | 458 | 0.763333 | 16.645646 | 60.493111 | 71.936195 | 213 | 245 | 142 | 28.943299 | 21.03545 | 0.379452 | 6.320186 | 3.397383 |
| mlp | 600 | 535 | 0.891667 | 16.647155 | 60.204883 | 70.081987 | 427 | 108 | 65 | 34.252324 | 21.477088 | 0.436755 | 0.415831 | 3.488327 |
| cnn | 600 | 485 | 0.808333 | 16.650717 | 60.46298 | 72.642648 | 191 | 294 | 115 | 29.349127 | 21.378486 | 1.959989 | 3.636117 | 3.733756 |

Ba lượt live khác đầu vào, không phải benchmark cùng video hoặc đo Accuracy có nhãn. Nhận xét thủ công còn chờ tại manual_observations.csv. Video recording declined by user; deployment chưa chốt. Không có video/ảnh webcam được lưu.
