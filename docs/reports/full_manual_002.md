# Báo cáo hệ thống và thực nghiệm HandGesture

Phiên full manual_002; dữ liệu và checkpoint đã kiểm chứng. Báo cáo tổng hợp kết quả offline, CPU classifier và ba lượt webcam thực tế. Benchmark cùng video và xác nhận thủ công đúng/sai từng cử chỉ còn chờ; chưa chốt model triển khai.

## Chương 1 Bài toán và thiết kế so sánh

Hệ thống nhận diện một bàn tay và sáu cử chỉ tĩnh: fist, one, peace, three, four, palm. Lớp thị giác được ánh xạ sang số0–5; fist/peace/palm tương ứng búa/kéo/bao ở tầng ứng dụng. Demo hiện chỉ nhận diện; trò chơi vẫn hoãn theo phạm vi nhóm đã thống nhất.

Hai biểu diễn đầu vào được so sánh trên cùng ID/split: landmark63 chiều cho RF/MLP và ROI RGB cho CNN. MediaPipe cung cấp21 điểm; tọa độ đổi sang pixel `(x*w,y*h,z*w)`, trừ cổ tay và chia khoảng cách3D lớn nhất tới cổ tay. ROI vuông cạnh1,4 lần cạnh bbox lớn nhất, padding đen vượt biên. Detector-input padding0,5, phiên bản MediaPipe0.10.21 và hash model đều được fingerprint; suy luận không lật đầu vào.

RF là tổ hợp300 cây giới hạn độ sâu20 và min_samples_leaf2. MLP63→128→64→6 dùng scaler chỉ fit train. CNN tự xây gồm GAP và spatial4×4→FC128; MobileNetV3-Small dùng pretrained ImageNet IMAGENET1K_V1, input224 và normalization tương ứng. Đây là mở rộng transfer learning, không phải cùng kiến trúc với CNN GAP.

## Chương 2 Dữ liệu và protocol

Nhóm đã thay phạm vi tự thu thập bằng Kaggle. 38.687 mẫu nguồn,24.755 được giữ và13.932 no_hand; tỷ lệ loại36,01%. Train/validation/test còn17.385/3.690/3.680. Mọi mẫu được giữ nguyên ID/split nguồn và mẫu loại có lý do. Cùng tập được giữ được dùng cho cả ba nhánh. Không có metadata người/phiên quay; split theo mẫu chưa chứng minh tổng quát hóa sang người mới.

Dữ liệu Kaggle nguồn đã crop, khác khung hình webcam thực tế. Fingerprint giống nhau không đủ để thay thế manifest: bản local lịch sử khác một số landmark/ROI, đã tách riêng; model full dùng data_v2 từ Kaggle.

Huấn luyện trên GPU T4, RF chạy CPU. MLP/CNN dùng validation macro-F1 rồi validation loss để chọn checkpoint; ba CNN cũng được chọn bằng cùng tiêu chí. Sau đó mới đánh giá test RF,MLP,CNN được chọn. Không đánh giá test các CNN thua để chọn lại cấu hình. Augmentation chỉ ở train, không flip. Môi trường, hash, config, history, validation/test prediction và report được lưu theo phiên.

## Chương 3 Kết quả và phân tích

### 3.1 Chất lượng offline

| model | architecture | test_accuracy | test_macro_f1 | validation_macro_f1 |
| --- | --- | --- | --- | --- |
| rf | rf | 0.916304 | 0.897465 | 0.898817 |
| mlp | mlp | 0.928261 | 0.910729 | 0.923535 |
| cnn | mobilenet_v3_small | 0.955707 | 0.946105 | 0.945704 |

MLP và MobileNet vượt mục tiêu Accuracy92%; RF chưa đạt. CNN cao hơn MLP khoảng2,74 điểm phần trăm trên tập test được giữ. Các con số này là chất lượng có điều kiện sau detector, không phải Accuracy trên toàn bộ ảnh nguồn hoặc webcam. Không tính mức cải thiện trực tiếp từ baseline lịch sử khác preprocessing/tập đánh giá.

CNN GAP/spatial/MobileNet lần lượt có validation macro-F1 0,754846/0,936817/0,945704; MobileNet được chọn. Không thay kết quả test bằng validation hoặc gọi MobileNet là CNN GAP ban đầu.

### 3.2 Classifier CPU

| model | cpu_latency_median_ms | cpu_latency_p95_ms | benchmark_samples | warmup | threads | batch_size |
| --- | --- | --- | --- | --- | --- | --- |
| rf | 7.592674 | 7.95466 | 200 | 20 | 1 | 1 |
| mlp | 0.091952 | 0.104796 | 200 | 20 | 1 | 1 |
| cnn | 3.366312 | 3.922955 | 200 | 20 | 1 | 1 |

Đo trên i7-1255U, một thread,batch1,20 warm-up,cùng200 mẫu cố định. MLP bao gồm scaler; chuẩn bị tensor CNN không thuộc classifier timing. MLP có classifier latency thấp nhất trong lượt đo. Chưa thể dùng nghịch đảo latency này làm FPS hệ thống. Thời gian train RF CPU và mạng GPU chỉ mô tả từng lượt, không phải benchmark tốc độ train cùng phần cứng.

### 3.3 Webcam trực tiếp

| model | measured_frames | hand_frames | fps | median_ms | p95_ms | recognized | uncertain | no_hand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf | 600 | 458 | 16.645646 | 60.493111 | 71.936195 | 213 | 245 | 142 |
| mlp | 600 | 535 | 16.647155 | 60.204883 | 70.081987 | 427 | 108 | 65 |
| cnn | 600 | 485 | 16.650717 | 60.46298 | 72.642648 | 191 | 294 | 115 |

Mỗi lượt warm-up30,đo600 frame,camera0,độ phân giải thực tế640×480,có overlay và cửa sổ hiển thị. Không lưu ảnh/video. Cả ba lượt hoàn tất frame limit và có tay thật. Tốc độ quan sát khoảng16,65FPS, vượt mốc15FPS trong từng lượt này; không tự công bố model triển khai vì đầu vào ba lượt khác nhau và thiếu xác nhận đúng/sai từng cử chỉ.

CSV xác nhận no_hand/uncertain không giữ nhãn, recognized có confidence≥0,7 và sau mất tay cần đủ frame tích lũy trước recognized. Các kiểm tra này không thay thế quan sát con người. Phím q/Esc chưa nghiệm thu vì ba lượt kết thúc tự động bằng frame limit. Số lần xuất hiện nhãn không phải số dự đoán đúng.

Bảng component timing và protocol nằm trong acceptance/live_comparison.csv và từng thư mục live_*_01. Mean component bao gồm frame không có tay nên không dùng để xếp hạng công bằng giữa ba lượt có tỷ lệ phát hiện khác nhau. Người dùng không đồng ý ghi clip, vì vậy phép so sánh cùng video vẫn chờ; không dựng video thay thế.

### 3.4 Phân tích lỗi và detector


Gallery được xem từ bản local, không public ảnh nguồn. Quan sát từng model cho thấy một số ROI bị mờ/độ phân giải thấp, góc tay nhìn cạnh hoặc ngón chồng nhau; các điều kiện này làm thông tin phân biệt khó quan sát. Đây là nhận xét trực quan, chưa phải kết luận nguyên nhân có đối chứng. Sáu mẫu CNN cần xem lại với ảnh nguồn Kaggle trước crop ROI và quy ước nhãn được lưu tại acceptance/error_review_cases.csv; không tự sửa nhãn test hoặc bỏ mẫu sau khi xem kết quả.

Các cặp nhầm phổ biến: RF peace→one35,three→four33,peace→three30; MLP one→fist38,peace→one32,three→peace29; CNN one→fist19,fist→one16,three→fist12. Độ gần hình học có thể là một giả thuyết nhưng tên cặp nhầm không đủ chứng minh landmark sai, crop sai hay nhãn sai.

Thống kê detector theo split/lớp nằm tại acceptance/detector_by_split_class.csv. no_hand là trạng thái trả về của pipeline, không phải detector Recall/Accuracy có ground truth;13.932 mẫu bị loại chưa có annotation đánh giá detector độc lập. Độ chính xác95,57% không bao gồm nhóm bị loại này.

### 3.5 Ảnh hưởng của detector đến toàn bộ tập nguồn

| gesture | source | accepted | rejected | rejected_pct |
| --- | --- | --- | --- | --- |
| fist | 6219 | 3950 | 2269 | 36.48 |
| four | 6679 | 5509 | 1170 | 17.52 |
| one | 6328 | 2302 | 4026 | 63.62 |
| palm | 6644 | 6332 | 312 | 4.7 |
| peace | 6430 | 3137 | 3293 | 51.21 |
| three | 6387 | 3525 | 2862 | 44.81 |

Tỷ lệ loại khác nhau theo lớp: one63,62%,palm4,70%. Tập được giữ vì vậy có phân bố lớp khác tập nguồn; macro-F1 và số mẫu từng lớp phải báo cùng Accuracy. Không suy ra nguyên nhân chỉ từ tỷ lệ này.

Chẩn đoán bổ sung trên toàn bộ5.804 mẫu test nguồn, theo nhãn nguồn và quy ước mọi no_hand là thất bại:

| model | source_test_samples | accepted_test_samples | detector_rejected_test_samples | classifier_correct | classifier_errors | success_fraction_source_test_no_hand_as_failure |
| --- | --- | --- | --- | --- | --- | --- |
| rf | 5804 | 3680 | 2124 | 3372 | 308 | 0.580979 |
| mlp | 5804 | 3680 | 2124 | 3416 | 264 | 0.58856 |
| cnn | 5804 | 3680 | 2124 | 3517 | 163 | 0.605961 |

Đây là phép đếm dẫn xuất từ split nguồn và prediction đã lưu, không chạy lại/chọn lại model, không sửa metrics gốc. CNN đúng3.517/3.680 mẫu được giữ nhưng chỉ3.517/5.804≈60,60% khi tính cả2.124 mẫu detector từ chối là thất bại. Chỉ số chẩn đoán này mô tả pipeline offline trên ảnh Kaggle đã crop theo quy ước trên; không phải Webcam Accuracy hoặc metric detector Recall có annotation. Nó cho thấy giới hạn của việc chỉ công bố95,57% classifier Accuracy.

## Kết luận hiện tại

Đã có đủ kết quả offline có thể kiểm chứng, classifier CPU và ba lượt webcam thực tế. MLP nhanh hơn ở classifier, CNN tốt hơn về test Accuracy trong protocol hiện tại. Chi phí capture/detector/render chiếm phần đáng kể của pipeline nên cần benchmark chung mới trả lời đầy đủ trade-off thời gian thực.

Còn chờ video chung (người dùng chọn chưa ghi), xác nhận thủ công sáu cử chỉ/phím thoát và lựa chọn deployment. Không thay người dùng xác nhận các mục chưa quan sát, không công bố Webcam Accuracy. Trò chơi vẫn nằm ngoài đợt này. Bản Markdown này là báo cáo kỹ thuật hiện trạng; bản trình bày học thuật theo mẫu trường và slide bảo vệ cần hoàn thiện sau khi các mục nghiệm thu được xác nhận.

## Nguồn bằng chứng trong repo

- Đề cương gốc giữ local, không public thông tin sinh viên.
- ../../reports/manual_002/final/comparison/report.md
- ../../reports/manual_002/benchmarks/cpu/classifier/classifier_protocol.json
- ../../reports/manual_002/acceptance/acceptance_status.json
- ../../reports/manual_002/dataset/preparation_report.json
