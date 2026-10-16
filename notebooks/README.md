# Notebook

| Vị trí | Vai trò |
|---|---|
| `kaggle_v2/07_interactive_workflow.ipynb` | Notebook chính để chạy từng bước trên Kaggle; smoke mặc định, có cờ full/test |
| `kaggle_v2/00`–`05` | Các bước v2 tách riêng: preprocessing, RF, MLP, CNN candidates, finalize, compare |
| `kaggle_v2/06_full_experiment.ipynb` | Chạy chuỗi tự động trong môi trường tương thích, không thay notebook thủ công chính |
| `kaggle/` | Notebook baseline lịch sử |
| `EDA.ipynb` | Khảo sát dữ liệu ban đầu, không phải kết quả full v2 |

Notebook phân phối sinh từ source `handgesture/` và `scripts/`, không chứa output. Không chép output full vào notebook phân phối: snapshot đã chạy nằm ở `full_manual_002.executed.ipynb` ở workspace local (có ảnh nguồn, không public). Snapshot này dùng để xem bằng chứng, không Run All lên cùng output path để train đè.

Sinh lại notebook bằng `scripts/build_kaggle_notebooks.py` và `scripts/build_v2_notebooks.py` từ gốc repo. Hướng dẫn full/kernel nằm ở README gốc.

Số liệu để viết báo cáo có ở [reports/manual_002](../reports/manual_002/README.md); chạy webcam dùng checkpoint GitHub Release như README gốc, không cần notebook đã thực thi hoặc dataset ảnh.
