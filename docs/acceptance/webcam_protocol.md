# Nghiệm thu webcam và benchmark end-to-end

Dùng checkpoint full `results_v2/kaggle_full_manual_002/experiment/final` và dữ liệu `data_v2`. Source đo đã có; không cần train lại để nghiệm thu. Không sửa metric hoặc dự đoán test gốc.

## Hai chế độ hiển thị

Webcam hỗ trợ `--mode count|rps`, mặc định count. Count chỉ hiện0–5; rps ánh xạ fist→Búa,peace→Kéo,palm→Bao, còn one/three/four hiện “Ngoài bộ cử chỉ”. Phím m đổi chế độ; q/Esc thoát. Confidence và trạng thái no_hand/uncertain vẫn hiển thị. Ghi rõ chế độ trong ghi chú nghiệm thu. CSV `label` luôn giữ lớp gốc để đối chiếu model, `display_mode` và `display_label` ghi nhãn UI của đúng frame. Đổi chế độ không đổi checkpoint hoặc chuẩn hóa xác suất trên ba lớp.

Các lượt live_01 dưới đây được đo trước khi tách overlay; cần lượt mới để nghiệm thu chế độ hiển thị và đo timing của overlay mới.

## Đã đo

Ba lượt camera0,640×480,warm-up30,600 frame đo, có cửa sổ hiển thị và overlay. Mỗi lượt mở/đóng camera riêng, chỉ lưu CSV timing và JSON protocol. Xem [báo cáo live](../../reports/manual_002/acceptance/report.md) và [báo cáo tổng hợp](../reports/full_manual_002.md).

Các lượt này dùng động tác thực hiện lại, không phải cùng một video. Chúng chứng minh pipeline chạy và đo tốc độ trong phiên này; không thay thế Accuracy có nhãn hoặc so sánh đầu vào chung. Người dùng đã chọn không ghi clip; benchmark cùng video còn chờ và không tự bật ghi hình.

## Checklist quan sát của người thử

Với từng RF/MLP/CNN, quan sát cả nhãn và trạng thái. Điền [manual_observations.csv](../../reports/manual_002/acceptance/manual_observations.csv), giữ trạng thái chờ với mục chưa thử. `observed` ghi đúng nhãn/trạng thái thực tế, `status` là `pass`, `fail` hoặc `not_tested`; `notes` ghi model, tư thế, khoảng cách, nền/ánh sáng hoặc cách tái hiện lỗi. Không ghi pass từ việc chỉ thấy model dự đoán một nhãn.

| Ca thử | Cách thử | Tiêu chí |
|---|---|---|
| Sáu cử chỉ | fist/one/peace/three/four/palm, giữ3–5giây | Recognized đúng nhãn khi bàn tay hiện rõ; ghi cả lỗi/không ổn định |
| Mất tay | Đưa tay ra khỏi khung | no_hand, không giữ nhãn cũ |
| Tư thế chưa rõ | Chuyển giữa hai tư thế hoặc xoay nghiêng | uncertain khi chưa đủ confidence/phiếu; không cố ép model phải uncertain ở mọi tư thế |
| Đưa tay trở lại | Một cử chỉ khác sau mất tay | Tích lũy lại đủ phiếu; không lấy nhãn lịch sử trước mất tay |
| Đổi model | Đóng lượt trước, chạy CLI với model khác | Tải đúng checkpoint/mapping, không còn tiến trình cũ giữ camera |
| Thoát | Bấm q hoặc Esc khi không đặt frame limit | Cửa sổ đóng, camera được giải phóng; lượt kết thúc bằng frame_limit không thay thử phím thoát |

Ví dụ chạy thử tự do (không ghi ảnh/video):

```bash
.venv-v2/bin/python scripts/webcam_demo.py --model mlp --mode count \
  --checkpoint-dir results_v2/kaggle_full_manual_002/experiment/final \
  --camera 0 --width 640 --height 480 \
  --output-dir results_v2/kaggle_full_manual_002/acceptance/live_mlp_manual_02
```

Đổi model/output-dir khi chuyển lượt. Kiểm tra độ phân giải thực tế trong frames.csv; giới hạn phần cứng, ánh sáng và người thực hiện phải ghi vào nhận xét. Không dùng tỷ lệ recognized làm Accuracy. Muốn đo Accuracy webcam cần nhãn ground truth và protocol riêng, tách uncertain/no_hand.

## Khi có video chung hợp lệ

Chỉ sử dụng video mà nhóm cung cấp hoặc đồng ý ghi rõ ràng. Cần ít nhất330 frame giải mã được:30 warm-up+300 đo, một tay hiện đủ nhiều (protocol deployment yêu cầu≥80% frame được detector phát hiện). Nên có đủ sáu tư thế, chuyển tay và khoảng cách/nền tương tự điều kiện demo. Tránh diễn giải bộ benchmark tốc độ này thành test tổng quát hóa.

```bash
.venv-v2/bin/python scripts/benchmark_v2.py \
  --results-root results_v2/kaggle_full_manual_002/experiment/final \
  --data-root data_v2 \
  --output-dir results_v2/kaggle_full_manual_002/benchmarks/common_video_01 \
  --video /path/to/hand-video.mp4 --display
```

Công cụ xử lý cùng video cho cả ba model và xuất CSV theo frame, protocol/hash video, FPS/median/p95, report và deployment.json. Chỉ chốt deployment khi đủ frame/có tay, FPS tổng và hand-positive FPS đạt15; trong các model đủ điều kiện chọn validation macro-F1 cao nhất rồi latency. Kèm nhận xét đúng/sai thực tế của người thử. Không tự đổi protocol để loại bỏ kết quả chậm hoặc chọn theo test Accuracy.

## Những xác nhận tự động đã làm được

Kiểm tra lại600 dòng mỗi model, resolution640×480, no_hand/uncertain không có label cũ, recognized có confidence≥0,7 và sau no_hand cần ít nhất5 frame trước recognized. Model checkpoint/fingerprint và hash log nằm trong acceptance_status.json. Đây là kiểm tra tính nhất quán của log; ground truth sáu cử chỉ và phím thoát vẫn cần người thử.
