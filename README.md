# AI Fitness Tracker 🏋️‍♂️

Ứng dụng đếm số lần tập hít đất (Push-up) và Squat sử dụng MediaPipe Pose và Machine Learning.

## Cài đặt

1. Cài đặt các thư viện yêu cầu:
```bash
pip install -r requirements.txt
```

2. Tạo dữ liệu mẫu:
```bash
python data/generate_sample_data.py
```

3. Huấn luyện mô hình:
```bash
python models/train_model.py
```

4. Chạy ứng dụng Streamlit:
```bash
streamlit run app.py
```

## Cấu trúc thư mục
- `app.py`: Ứng dụng giao diện chính (Streamlit)
- `models/`: Chứa script huấn luyện và các tiện ích liên quan đến mô hình ML
- `utils/`: Chứa các hàm xử lý MediaPipe và xử lý dữ liệu
- `data/`: Chứa script tạo dữ liệu mẫu và lưu trữ dữ liệu
