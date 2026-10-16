# Công cụ chạy từ gốc repo

Dùng `.venv-v2/bin/python scripts/<tên>.py`. Các CLI được giữ ở vị trí hiện tại để không phá lệnh đã dùng/notebook/test. Thư viện thuật toán chính nằm trong `handgesture/`.

| Nhóm | Script |
|---|---|
| Thực nghiệm v2 | `prepare_data_v2.py`, `train_v2.py`, `benchmark_v2.py`, `webcam_demo.py` |
| Source notebook Kaggle | `build_interactive_notebook.py`, `build_v2_notebooks.py` |
| Smoke/kiểm tra môi trường | `smoke_test_v2.py`, `check_runtime.py` |
| Weights pretrained | `download_pretrained.py` |
| Kaggle API bổ trợ | `submit_kaggle_v2.py`, `fetch_kaggle_v2.py`, `watch_kaggle_v2.py` |
| Baseline lịch sử | `prepare_data.py`, `reorganize_data.py`, `normalize_results.py`, `build_kaggle_notebooks.py`, `smoke_test_kaggle_notebooks.py` |
| Reporting chung được nhúng vào notebook | `kaggle_notebook_support.py` |

Notebook chính là `notebooks/kaggle_v2/07_interactive_workflow.ipynb`. Bộ `00`–`06` vẫn phục vụ tái lập/chạy tách bước. Không dùng normalize_results cho v2.

Output full hiện hành: `results_v2/kaggle_full_manual_002/experiment/final`; dữ liệu: `data_v2`. Luôn truyền `--results-root`/`--checkpoint-dir` rõ ràng; mặc định `results_v2/final` của một số CLI là ví dụ tổng quát, không phải đường dẫn phiên này.

API/watcher là workflow phụ, không đại diện phiên tương tác hiện tại. Công cụ fetch cũ nhắm archive ở root của wrapper API; không dùng nguyên trạng để tải output có thư mục lồng của notebook tương tác. Xem hồ sơ retrieval của manual_002 thay cho chạy lại watcher lịch sử.
