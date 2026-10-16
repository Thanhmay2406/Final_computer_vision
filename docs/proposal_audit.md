# Đối chiếu source và kết quả với đề cương

Kiểm tra ngày 16/10/2026 đối với phiên full `manual_002`. Phần offline đã hoàn tất và kiểm chứng; hệ thống chưa đủ bằng chứng để kết luận chạy webcam thời gian thực hoặc chốt model triển khai. Benchmark classifier CPU đã được bổ sung trong lần rà soát này.

Tài liệu tham chiếu: đề cương gốc của nhóm (giữ local, không public thông tin sinh viên), đặc biệt các mục 2.1, 2.2, 3.4, 5.2–5.4 và bố cục Chương 3. Không sửa đề cương. Yêu cầu dữ liệu tự thu thập 1.500–2.000 mẫu được xem là **thay đổi phạm vi đã thống nhất sang Kaggle**, không phải yêu cầu phải thu thập lại trong đợt này. Trò chơi được nhóm hoãn; vẫn ghi nhận phần đó khi đối chiếu nguyên văn đề cương.

## Cập nhật nghiệm thu sau audit

Đã bổ sung [báo cáo hệ thống/thực nghiệm](reports/full_manual_002.md), phân tích trực quan gallery, thống kê detector theo split/lớp và ba lượt webcam thật600 frame ở640×480. Các lượt đạt khoảng16,65FPS; kiểm tra CSV trạng thái đã thực hiện. Xem [quy trình nghiệm thu](acceptance/webcam_protocol.md). Người dùng từ chối ghi clip nên benchmark cùng video vẫn chờ. Nhận xét đúng/sai sáu cử chỉ và thử phím thoát chưa được người quan sát xác nhận; không coi ba lượt live là đã hoàn tất mọi mục dưới đây.

## Những phần đã có

| Yêu cầu trong đề cương | Bằng chứng trong repo | Đánh giá |
|---|---|---|
| Một bàn tay, sáu lớp số đếm 0–5, nhãn không mâu thuẫn với kéo/búa/bao (2.2, 3.1, 3.4) | `handgesture/data.py`, mapping trong checkpoint và metadata; quy ước ứng dụng ở README/metadata nguồn | Đủ lớp thị giác. Tầng demo chưa có chế độ số đếm/trò chơi hoặc phát lệnh ứng dụng |
| MediaPipe 21 điểm, chuẩn hóa theo cổ tay và kích thước tay (2.2, 5.2) | `handgesture/preprocessing.py`; model hash, MediaPipe version và fingerprint trong dataset/checkpoint | Có và có test geometry/invalid/no_hand |
| ROI RGB, resize/normalize, augmentation chỉ train (2.2, 5.2) | `handgesture/models.py`, `training.py`; hai CNN 128×128 và MobileNet 224×224 | Có; CNN không nhận landmark vector |
| RF, MLP, CNN cùng split/mapping/protocol (5.3) | `experiment/runs/`, `experiment/candidates/`, `final/selection.json` | Đủ năm lượt train, CNN chọn bằng validation; test chỉ ba model đã chốt |
| Validation, early stopping và checkpoint selection (5.3) | History, validation_predictions, model_config/run.json của MLP và ba CNN | Có, kiểm chứng lại từ CSV/hash |
| Accuracy/Precision/Recall/F1/macro và confusion matrix (5.4) | Final từng model, `comparison/` | Có, tính lại được từ CSV dự đoán; cùng 3.680 mẫu test |
| Mục tiêu Accuracy ≥92% (2.1, 5.4) | RF 91,6304%; MLP 92,8261%; MobileNet 95,5707% | MLP và CNN đạt; RF chưa đạt. Không khẳng định mọi model đều đạt |
| Thời gian train và môi trường (2.2, Chương 3.1/3.7) | `run.json`, `metrics.json`, `environment.json` và bảng comparison | Có. RF train CPU, mạng train GPU nên không dùng thời gian này để kết luận train nhanh hơn trên cùng phần cứng |
| Latency riêng classifier trên cùng máy (5.4) | `benchmarks/cpu/classifier/` | Đã đo bổ sung: CPU một thread, batch1, 20 warm-up, cùng 200 mẫu test seed42, median/p95 |
| Pipeline nhận diện webcam (2.1, 3.4) | `scripts/webcam_demo.py`, `handgesture/live.py` | Source có, dùng full checkpoint được; chưa nghiệm thu thực tế sáu cử chỉ |
| Phân tích nhầm lớp và trường hợp sai (5.4, Chương 3.8) | Per-class metrics, top_confusions, gallery theo lớp và ba cặp nhầm/model | Có số liệu/ảnh. Còn cần diễn giải nguyên nhân và xác nhận thủ công |
| Source module, dependency, README, khả năng tái lập (5.5) | `handgesture/`, `scripts/`, `tests/`, `requirements/`, notebook generator | Có; source không đổi thuật toán trong lần dọn này |

Các đường dẫn `experiment/`, `final/`, `benchmarks/` trong báo cáo này thuộc `results_v2/kaggle_full_manual_002/`.

## Kết quả đã xác minh

| Model cuối | Validation macro-F1 | Test Accuracy | Test macro-F1 | CPU median (ms) | CPU p95 (ms) |
|---|---:|---:|---:|---:|---:|
| RF | 0,898817 | 91,6304% | 0,897465 | 7,592674 | 7,954660 |
| MLP | 0,923535 | 92,8261% | 0,910729 | 0,091952 | 0,104796 |
| CNN MobileNetV3-Small | 0,945704 | 95,5707% | 0,946105 | 3,366312 | 3,922955 |

CPU: Intel Core i7-1255U; Python3.11.17, Torch2.11.0+cpu, sklearn1.6.1. Nguồn: [protocol](../reports/manual_002/benchmarks/cpu/classifier/classifier_protocol.json) và [latency CSV](../reports/manual_002/benchmarks/cpu/classifier/classifier_latency.csv). MLP gồm scaler; chuẩn bị tensor CNN không nằm trong classifier timing. Phép đo này không gồm decode, MediaPipe, crop hay render. Không đổi nghịch đảo latency thành FPS webcam hoặc dùng nó để công bố đạt ≥15 FPS.

MobileNet là ứng viên transfer learning bổ sung so với CNN nhẹ ban đầu. CNN GAP đạt validation Accuracy 78,5637%, macro-F1 0,754846; CNN spatial đạt validation Accuracy 94,9322%, macro-F1 0,936817; MobileNet đạt validation Accuracy 95,4743%, macro-F1 0,945704. Các ứng viên thua không được đánh giá test theo protocol đã khóa; thiếu test của GAP/spatial không phải artifact bị bỏ sót. Báo cáo phải nêu rõ CNN final là pretrained MobileNet, không gọi nó là kiến trúc GAP ban đầu.

Trong kết quả hiện tại, CNN có test Accuracy cao hơn MLP khoảng 2,74 điểm phần trăm, còn MLP có classifier latency thấp hơn. Đây chưa phải câu trả lời đầy đủ cho trade-off end-to-end vì chi phí detector/preprocessing chưa được đo trên cùng video/webcam. Không dùng khác biệt này để tự khẳng định ưu thế có ý nghĩa thống kê hoặc khả năng tổng quát hóa sang người mới.

## Những phần còn thiếu và thứ tự hoàn tất

1. **Benchmark end-to-end trên CPU, cùng video cho cả ba model.** Source đã có `benchmark_video`, CSV capture/decode/detector/preprocessing/classifier/render và bộ chọn deployment. Chưa có output full cho phép đo này; nhóm xác nhận chưa có video chung. Cần ít nhất330 frame (30 warm-up+300 đo), đủ frame có tay, cùng độ phân giải và điều kiện render. Đo median/p95/FPS, ghi protocol máy. Không tự tạo video tổng hợp để thay nghiệm thu bàn tay thật.
2. **Nghiệm thu webcam trực tiếp 640×480.** Source và test camera giả có; probe camera cũ không có tay nên không chứng minh nhận diện đúng. Cần kiểm tra sáu cử chỉ, mất tay, confidence thấp, chuyển model, thoát/giải phóng camera và điều kiện ánh sáng/nền mới. Lưu timing CSV/protocol, ghi kết quả quan sát; không cần tự lưu video webcam.
3. **Lựa chọn model triển khai và kết luận trade-off.** Chưa có `deployment.json` đủ điều kiện từ full video. Mục tiêu ≥15 FPS theo kế hoạch triển khai hiện hành; trong model đạt, chọn validation macro-F1 rồi latency. CNN thắng offline không tự động là lựa chọn webcam. Nếu không model đạt, ghi rõ chưa đạt.
4. **Phân tích detector và lỗi thực nghiệm bằng diễn giải.** Có danh sách13.932 mẫu bị loại đều `no_hand`, gallery và các cặp nhầm. Cần xem mẫu cụ thể để phân biệt detector bỏ sót, ảnh nguồn đã crop, blur/góc tay và nhãn gần nhau; chưa có đối chứng/annotation để gọi tỷ lệ `no_hand` là detector Accuracy/Recall. Không gán nguyên nhân chỉ dựa vào tên cặp nhầm.
5. **Báo cáo cuối và slide nếu nghiệm thu toàn bộ sản phẩm đề cương.** Repo có proposal, tài liệu pipeline, report kỹ thuật từng model và notebook; chưa thấy báo cáo đồ án hoàn chỉnh Chương1–3/kết luận hoặc slide thuyết trình. Chương3 cần hợp nhất chất lượng, detector rejection, CPU latency, end-to-end và giới hạn dữ liệu. Không sửa đề cương trong tác vụ này.
6. **Demo ứng dụng/trò chơi theo bản đề cương gốc.** Hiện chỉ có webcam nhận diện/overlay; chưa có logic trò chơi, ánh xạ sang lệnh điều khiển hoặc chuyển chế độ số đếm/kéo-búa-bao. Đây là phần được nhóm hoãn theo phạm vi đã thống nhất trước đó, nên chưa xây thêm ở đợt dọn repo.

Lệnh benchmark/webcam cho checkpoint hiện hành nằm trong [README gốc](../README.md). Ở thời điểm audit ban đầu chưa chạy camera. Sau đó đã chạy ba lượt live có người thực hiện; xem mục cập nhật ở đầu tài liệu. Vẫn chưa có video chung để chạy benchmark đầu vào giống nhau.

## Những khác biệt cần ghi đúng trong báo cáo

- **Nguồn/quy mô dữ liệu:** đề cương gốc mô tả tự thu thập1.500–2.000 mẫu, không trộn public data. Nhóm đã đổi sang Kaggle và giữ split hiện tại. Không mô tả lại dữ liệu này thành bộ tự thu thập; không yêu cầu quay lại kế hoạch cũ trong đợt này.
- **Subject/session split:** đề cương5.2 ưu tiên tách người/phiên quay; dataset hiện không có metadata đó. Split đang theo mẫu, giữ nguyên ID và không tự chia lại. Đây là giới hạn khoa học chưa đáp ứng nguyên văn đề cương, không thể sửa bằng đổi nhãn file hoặc tuyên bố test trên người mới.
- **Metric phụ thuộc detector:** 38.687 mẫu nguồn →24.755 được giữ (63,99%),13.932 bị loại (36,01%). Split giữ lại17.385/3.690/3.680. Test Accuracy95,57% của CNN chỉ tính trên3.680 mẫu test được giữ, không phản ánh toàn bộ ảnh nguồn hoặc webcam. Dữ liệu Kaggle đã crop và detector còn dùng padding đầu vào được fingerprint; cần nêu giới hạn miền ảnh.
- **Tinh chỉnh RF/MLP:** có một cấu hình full RF và một full MLP; chưa thấy bảng tìm kiếm nhiều cấu hình RF để chứng minh phần “tinh chỉnh số cây/độ sâu/max_features” của5.3. Có thể báo cáo đây là cấu hình cố định đã thống nhất. Nếu bổ sung nghiên cứu hyperparameter, chọn bằng validation và ghi phiên riêng; không dùng kết quả test hiện tại để săn cấu hình tốt hơn.
- **CNN pretrained:** đã ghi weights/hash và normalization; là mở rộng có chủ đích. Cần giữ kết quả CNN tự xây trên validation để giải thích lựa chọn, không gộp mọi kiến trúc thành một CNN duy nhất.
- **Tách lịch sử:** baseline ở `results/` dùng preprocessing/tập đánh giá khác; không tính phần trăm cải thiện trực tiếp từ đó sang full mới. Bản local preprocessing cũ khác manifest Kaggle, nay ở `data_v2_local_archive/`; không dùng với final checkpoint.

## Các điểm kỹ thuật nên cải thiện sau nghiệm thu

RF hiện báo bắt đầu/kết thúc nhưng `RandomForestClassifier.fit()` không phát tiến độ từng cây. MLP/CNN báo batch đầu, mỗi25 batch và cuối batch, cùng epoch metric. Dataset validation/hash scan và lượt đánh giá checkpoint tốt nhất còn có khoảng im lặng. Đây là hạn chế quan sát đã xác nhận từ source, không phải bằng chứng model bị treo; chưa sửa behavior của phiên train đã hoàn tất.

Công cụ API fetch cũ hướng tới archive ở root của wrapper, trong khi notebook tương tác đặt archive dưới thư mục phiên. Vì vậy giữ nó như workflow phụ và không coi trạng thái watcher lịch sử là trạng thái full tương tác. Notebook/API/watcher cũ được phân loại trong [scripts/README.md](../scripts/README.md).
