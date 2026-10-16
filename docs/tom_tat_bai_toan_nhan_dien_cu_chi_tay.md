# TÓM TẮT BÀI TOÁN NHẬN DIỆN CỬ CHỈ TAY

> Tài liệu này tóm tắt đề cương ban đầu. Nhóm đã thống nhất đổi sang dữ liệu Kaggle và tạm hoãn trò chơi; trạng thái source/kết quả thực tế nằm ở [README hiện hành](../README.md) và [báo cáo đối chiếu](proposal_audit.md).

**Tên đề tài:** Xây dựng hệ thống nhận diện cử chỉ tay đơn giản bằng MediaPipe và CNN

**Tên tiếng Anh:** *HandGesture: A Real-time Hand Gesture Recognition System*

**Lĩnh vực:** Thị giác máy tính (Computer Vision), học máy (Machine Learning), tương tác người–máy (Human–Computer Interaction, HCI).

## 1. Phát biểu bài toán

Xây dựng một hệ thống nhận diện **cử chỉ tay tĩnh** từ ảnh RGB hoặc khung hình webcam, có khả năng đưa ra dự đoán cử chỉ theo thời gian thực và ánh xạ kết quả nhận diện thành thao tác điều khiển trong một ứng dụng hoặc trò chơi.

Bài toán không chỉ yêu cầu xây dựng một mô hình phân loại, mà còn thực hiện **so sánh thực nghiệm có kiểm soát** giữa hai hướng biểu diễn đầu vào:

- **Landmark-based:** sử dụng cấu trúc hình học bàn tay được biểu diễn bằng 21 điểm mốc (*hand landmarks*).
- **Image-based:** sử dụng trực tiếp vùng ảnh RGB chứa bàn tay để mô hình học đặc trưng thị giác.

Mục đích cuối cùng là lựa chọn phương pháp phù hợp cho hệ thống hoạt động trên máy tính cá nhân phổ thông, không bắt buộc GPU chuyên dụng.

## 2. Đầu vào và đầu ra

### Đầu vào

- Ảnh RGB hoặc các khung hình lấy từ webcam.
- Mỗi khung hình tập trung vào **một bàn tay** đang thực hiện một cử chỉ tĩnh.
- Tùy nhánh xử lý, đầu vào của bộ phân loại là:
  - Vector tọa độ của **21 hand landmarks**, mỗi điểm có các thành phần `(x, y, z)` và được chuẩn hóa; hoặc
  - Ảnh RGB của vùng bàn tay (*region of interest*, ROI) đã được crop, resize và chuẩn hóa.

### Đầu ra

- **Nhãn cử chỉ** mà hệ thống dự đoán.
- Kết quả có thể được chuyển thành **lệnh điều khiển** trong ứng dụng/trò chơi.
- Trong quá trình thực nghiệm, hệ thống còn ghi nhận các chỉ số về chất lượng phân loại và tốc độ xử lý.

## 3. Tập cử chỉ và quy ước nhãn

Bài toán tập trung vào các cử chỉ tay tĩnh đơn giản:

- Các tư thế số đếm từ **0 đến 5**.
- Các tư thế phục vụ trò chơi **kéo – búa – bao**.

**Nguyên tắc quan trọng:** Nếu hai ý nghĩa ứng dụng tương ứng với cùng một tư thế bàn tay (chẳng hạn tư thế số 2 và kéo), hệ thống chỉ học **một lớp thị giác** cho tư thế đó. Ý nghĩa cụ thể sẽ được ánh xạ theo chế độ sử dụng của ứng dụng, tránh gán hai nhãn mâu thuẫn cho cùng một dạng hình ảnh.

## 4. Các phương pháp được so sánh

### 4.1. MediaPipe Hands + Random Forest

1. Phát hiện bàn tay bằng MediaPipe Hands.
2. Trích xuất **21 hand landmarks**.
3. Chuẩn hóa tọa độ theo vị trí cổ tay và kích thước bàn tay; chuyển thành vector đặc trưng.
4. Sử dụng **Random Forest** để phân loại cử chỉ.

Phương pháp này kiểm tra khả năng nhận diện của một bộ phân loại dựa trên cây khi chỉ sử dụng đặc trưng hình học bàn tay.

### 4.2. MediaPipe Hands + Multi-Layer Perceptron (MLP)

1. Phát hiện bàn tay và trích xuất landmark tương tự nhánh Random Forest.
2. Sử dụng cùng cách chuẩn hóa và biểu diễn vector đầu vào.
3. Huấn luyện **MLP** để học quan hệ phi tuyến giữa các tọa độ khớp tay và dự đoán cử chỉ.

Phương pháp này giúp đối chiếu mô hình mạng nơ-ron truyền thẳng với Random Forest trên **cùng loại đặc trưng landmark**.

### 4.3. Ảnh RGB + CNN nhẹ

1. Xác định vùng bàn tay và **crop ROI** từ ảnh RGB (MediaPipe có thể hỗ trợ xác định ROI).
2. Resize ảnh về kích thước cố định và chuẩn hóa giá trị pixel.
3. Sử dụng **CNN nhẹ** để trực tiếp học các đặc trưng không gian và phân loại cử chỉ.

Trong thiết kế so sánh chính, CNN nhận **ảnh RGB**, không nhận vector landmark. Nhánh này đóng vai trò đối chứng với hai phương pháp landmark-based.

## 5. Pipeline xử lý tổng quát

```text
Ảnh RGB / Webcam
        |
        v
 Phát hiện vùng bàn tay
        |
        +-----------------------------+
        |                             |
        v                             v
 NHÁNH LANDMARK-BASED           NHÁNH IMAGE-BASED
 Trích xuất 21 landmarks         Crop vùng ảnh bàn tay
        |                             |
 Chuẩn hóa tọa độ                Resize + chuẩn hóa ảnh
        |                             |
        +-------------+               v
        |             |            CNN nhẹ
        v             v               |
   Random Forest     MLP              |
        |             |               |
        +-------------+---------------+
                      |
                      v
             Dự đoán nhãn cử chỉ
                      |
                      v
        Đánh giá và so sánh các mô hình
                      |
                      v
        Chọn phương pháp cho demo thời gian thực
                      |
                      v
         Ánh xạ cử chỉ thành lệnh điều khiển
```

Ba mô hình được huấn luyện độc lập và đánh giá theo **cùng quy ước lớp, cùng protocol thử nghiệm và cùng điều kiện phần cứng** để bảo đảm phép so sánh nhất quán.

## 6. Mục tiêu nghiên cứu

1. Xây dựng pipeline nhận diện cử chỉ tay tĩnh từ webcam.
2. Triển khai và so sánh ba phương pháp:
   - **Landmark + Random Forest**;
   - **Landmark + MLP**;
   - **Ảnh RGB + CNN nhẹ**.
3. Đánh giá chất lượng nhận diện bằng **Accuracy, Precision, Recall, F1-score và Confusion Matrix**.
4. Đánh giá hiệu năng tính toán bằng **thời gian huấn luyện, độ trễ của riêng bộ phân loại, độ trễ end-to-end và FPS**.
5. Phân tích các trường hợp nhận diện sai, chẳng hạn cử chỉ có hình dạng gần giống nhau, góc quan sát bất lợi hoặc lỗi phát hiện bàn tay/crop ảnh.
6. Lựa chọn mô hình cân bằng tốt giữa **độ chính xác** và **khả năng vận hành thời gian thực** để tích hợp vào ứng dụng/trò chơi.

**Mục tiêu chất lượng theo đề cương:** Accuracy trên tập test đạt **từ 92% trở lên**. Đây là mục tiêu kỳ vọng, không phải kết quả thực nghiệm đã được chứng minh.

## 7. Tiêu chí đánh giá

| Nhóm tiêu chí | Chỉ số | Ý nghĩa |
|---|---|---|
| Chất lượng phân loại | Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| Chất lượng phân loại | Precision, Recall, F1-score | Chất lượng nhận diện từng lớp; quan tâm cả macro-average |
| Phân tích lỗi | Confusion Matrix | Xác định các cặp cử chỉ thường bị nhầm lẫn |
| Chi phí huấn luyện | Training time | Thời gian huấn luyện của từng phương pháp |
| Tốc độ bộ phân loại | Classifier latency | Thời gian chỉ tính bước dự đoán của RF/MLP/CNN |
| Tốc độ toàn hệ thống | End-to-end latency | Thời gian từ khung hình đầu vào đến nhãn dự đoán |
| Khả năng thời gian thực | End-to-end FPS | Số khung hình toàn hệ thống xử lý mỗi giây |

Kết luận không chỉ dựa vào Accuracy: một mô hình chính xác hơn nhưng chậm hơn đáng kể có thể không phải lựa chọn thích hợp nhất cho demo thời gian thực.

## 8. Phạm vi và giới hạn bài toán

**Trong phạm vi:**

- Phân loại **cử chỉ tay tĩnh** theo từng khung hình.
- Sử dụng **camera RGB thông thường** và một bàn tay trong khung hình.
- So sánh các phương pháp RF, MLP và CNN nhẹ.
- Triển khai demo thời gian thực trên phần cứng phổ thông.

**Ngoài phạm vi chính:**

- Cử chỉ động cần phân tích chuỗi thời gian.
- Nhận diện ngôn ngữ ký hiệu liên tục.
- Nhận diện đồng thời nhiều bàn tay hoặc nhiều người.
- Các trường hợp che khuất nghiêm trọng hoặc ánh sáng quá kém.
- Đề xuất kiến trúc học sâu hoàn toàn mới.

## 9. Câu hỏi nghiên cứu trung tâm

> **Đối với các cử chỉ tay tĩnh đơn giản, phân loại dựa trên hand landmarks có thể đạt độ chính xác tương đương hoặc tốt hơn, đồng thời có tốc độ và chi phí tính toán thuận lợi hơn so với phân loại trực tiếp trên ảnh RGB bằng CNN nhẹ hay không?**

Để trả lời, cần so sánh **Landmark + Random Forest**, **Landmark + MLP** và **Ảnh RGB + CNN nhẹ** trên cùng một protocol, đo cả chất lượng dự đoán lẫn hiệu năng end-to-end.

## 10. Sản phẩm đầu ra dự kiến

- Pipeline phát hiện bàn tay, trích xuất landmark và xử lý ảnh RGB.
- Ba mô hình phân loại cử chỉ tay đã được huấn luyện.
- Bảng và biểu đồ so sánh chất lượng nhận diện, độ trễ và FPS.
- Phân tích lỗi và lập luận lựa chọn mô hình phù hợp.
- Demo ứng dụng/trò chơi nhận diện và điều khiển bằng cử chỉ tay theo thời gian thực.
- Source code, hướng dẫn chạy và báo cáo thực nghiệm.

## 11. Tóm tắt ngắn gọn

Đề tài xây dựng hệ thống nhận diện cử chỉ tay tĩnh từ webcam và so sánh **hai hướng tiếp cận: đặc trưng hình học từ MediaPipe Hands và đặc trưng hình ảnh do CNN học trực tiếp**. Ba phương pháp được kiểm nghiệm gồm **MediaPipe + Random Forest**, **MediaPipe + MLP** và **RGB + CNN nhẹ**. Các mô hình được đánh giá không chỉ theo độ chính xác mà còn theo độ trễ và FPS, nhằm chọn giải pháp phù hợp cho ứng dụng nhận diện cử chỉ tay **theo thời gian thực trên máy tính phổ thông**.
