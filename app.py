import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import asyncio

from utils.pose_utils import PoseProcessor
from utils.data_utils import load_real_data, save_video_data
from models.model_utils import load_model_cached, load_metrics, predict_phase
from models.train_model import train_models

# Suppress Streamlit coroutine warning
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Page config
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED FUNCTIONS ---
def get_dataset():
    return load_real_data()

@st.cache_resource
def get_model():
    return load_model_cached()

def save_workout_history(exercise_type, reps, duration, accuracy=None):
    """Save workout session to history."""
    history_file = 'data/workout_history.csv'
    os.makedirs('data', exist_ok=True)
    
    new_record = pd.DataFrame([{
        'date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'exercise': exercise_type,
        'reps': reps,
        'duration_seconds': round(duration, 1),
        'accuracy': accuracy if accuracy is not None else 'N/A'
    }])
    
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, new_record], ignore_index=True)
    else:
        history_df = new_record
        
    history_df.to_csv(history_file, index=False)

def extract_frames_from_video(video_path, exercise_type, phase_label):
    cap = cv2.VideoCapture(video_path)
    processor = PoseProcessor()
    angles_list = []
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = processor.extract_keypoints(frame)
        angles = processor.get_exercise_angles(landmarks)
        
        if angles:
            angles_list.append(angles)
            
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
    cap.release()
    
    if angles_list:
        saved_count = save_video_data(angles_list, exercise_type, phase_label)
        return True, saved_count
    return False, 0

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/color/96/000000/dumbbell.png", width=80)
st.sidebar.title("AI Fitness Tracker")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Điều hướng",
    [
        "📊 1. Giới thiệu & Khám phá dữ liệu (EDA)", 
        "🏋️ 2. Triển khai mô hình (Demo)", 
        "📈 3. Đánh giá & Hiệu năng", 
        "📅 4. Lịch sử Tập luyện", 
        "📄 5. Báo cáo chi tiết"
    ]
)

st.sidebar.markdown("---")

model, scaler, le = get_model()
model_trained = model is not None

if not model_trained and page != "📊 1. Giới thiệu & Khám phá dữ liệu (EDA)":
    st.warning("⚠️ Mô hình chưa được huấn luyện! Vui lòng thu thập dữ liệu và huấn luyện trước.")

# ==========================================
# PAGE 1: GIỚI THIỆU & EDA
# ==========================================
if page == "📊 1. Giới thiệu & Khám phá dữ liệu (EDA)":
    st.markdown('<p class="main-header">Giới thiệu & Khám phá dữ liệu (EDA)</p>', unsafe_allow_html=True)
    
    st.info("""
    **Đề tài:** Tự động đếm số lần tập hít đất và squat từ video người tập bằng MediaPipe Pose kết hợp học máy nhằm hỗ trợ luyện tập thể dục tại nhà hiệu quả  
    **Sinh viên:** Nguyễn Văn A  
    **MSSV:** 20201234
    """)
    
    st.write("""
    **Giá trị thực tiễn:** Ứng dụng giúp người dùng tự tập luyện tại nhà hiệu quả hơn bằng cách tự động đếm số lần tập và đưa ra nhắc nhở về tư thế theo thời gian thực. Bằng việc kết hợp Computer Vision (MediaPipe) và Machine Learning (Random Forest), hệ thống có thể chạy nhẹ nhàng trên các thiết bị cá nhân mà không cần cấu hình cao.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Khám phá dữ liệu (EDA)", "📥 Thu thập Dữ liệu", "🧠 Huấn luyện Mô hình"])
    
    with tab1:
        st.write("### 🔍 Phân tích Dữ liệu Thực tế (EDA)")
        df = get_dataset()
        
        if df is not None and not df.empty:
            st.write("**1. Dữ liệu thô (Raw Data):**")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write("**2. Biểu đồ phân tích:**")
            col1, col2 = st.columns(2)
            with col1:
                counts = df['phase_label'].value_counts().reset_index()
                counts.columns = ['phase_label', 'count']
                fig1 = px.bar(counts, x='phase_label', y='count', color='phase_label',
                             title="Phân phối Nhãn (Class Distribution)")
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                numeric_df = df[['angle_elbow', 'angle_shoulder', 'angle_hip', 'angle_knee']]
                corr = numeric_df.corr()
                fig2 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Greens',
                                title="Ma trận Tương quan (Correlation Matrix)")
                st.plotly_chart(fig2, use_container_width=True)
                
            st.write("**Phân bố Đặc trưng (Feature Distribution):**")
            fig3 = px.box(df, x="phase_label", y="angle_elbow", color="phase_label", title="Phân bố Góc Khuỷu tay theo Nhãn")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("""
            **📝 Giải thích & Nhận xét:**
            - **Phân phối nhãn:** Biểu đồ cột cho thấy sự cân bằng (hoặc mất cân bằng) giữa các pha Lên/Xuống. Nếu dữ liệu bị lệch (imbalanced), mô hình có xu hướng dự đoán thiên về lớp đa số.
            - **Ma trận tương quan:** Thể hiện mối quan hệ tuyến tính giữa các góc khớp. Ví dụ: trong bài Squat, góc hông và góc gối thường có độ tương quan thuận rất cao.
            - **Phân bố đặc trưng:** Biểu đồ Boxplot cho thấy sự khác biệt rõ rệt của góc khuỷu tay giữa pha `push-up_up` và `push-up_down`, chứng tỏ đây là đặc trưng quan trọng nhất để phân loại.
            """)
        else:
            st.info("Tập dữ liệu đang trống. Vui lòng chuyển sang tab 'Thu thập Dữ liệu' để tải lên video.")

    with tab2:
        st.write("Tải lên các đoạn video ngắn (chỉ chứa pha LÊN hoặc XUỐNG) để xây dựng tập dữ liệu huấn luyện.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            ex_type = st.selectbox("Loại bài tập", ["Push-up", "Squat"])
            phase = st.selectbox("Pha chuyển động (Label)", [f"{ex_type.lower()}_up", f"{ex_type.lower()}_down"])
            uploaded_train_vid = st.file_uploader("Chọn video huấn luyện", type=['mp4', 'avi', 'mov'], key="train_vid")
            
        with col2:
            if uploaded_train_vid is not None:
                if st.button("Trích xuất & Lưu vào Dataset", type="primary"):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                    tfile.write(uploaded_train_vid.read())
                    
                    st.write("Đang trích xuất đặc trưng...")
                    success, count = extract_frames_from_video(tfile.name, ex_type, phase)
                    
                    if success:
                        st.success(f"✅ Đã trích xuất và lưu {count} frames với nhãn '{phase}'.")
                    else:
                        st.error("❌ Không tìm thấy người trong video hoặc có lỗi xảy ra.")

    with tab3:
        st.write("Huấn luyện mô hình Machine Learning (Random Forest) từ tập dữ liệu đã thu thập.")
        
        df = get_dataset()
        if df is not None and len(df) > 100:
            st.success(f"Tập dữ liệu hiện có {len(df)} frames. Đã sẵn sàng để huấn luyện!")
            
            st.write("### ⚙️ Tùy chỉnh Siêu tham số (Hyperparameters)")
            col_hp1, col_hp2, col_hp3 = st.columns(3)
            with col_hp1:
                n_estimators = st.slider("Số lượng cây (n_estimators)", 50, 300, 100, 50)
            with col_hp2:
                max_depth_opt = st.selectbox("Độ sâu tối đa (max_depth)", ["Không giới hạn", 10, 20, 30])
                max_depth = None if max_depth_opt == "Không giới hạn" else max_depth_opt
            with col_hp3:
                use_aug = st.checkbox("Sử dụng Data Augmentation (Thêm nhiễu)", value=True)
                
            if st.button("🚀 Bắt đầu Huấn luyện", type="primary", use_container_width=True):
                with st.spinner("Đang huấn luyện mô hình..."):
                    success, msg = train_models(n_estimators=n_estimators, max_depth=max_depth, use_augmentation=use_aug)
                    if success:
                        st.success(msg)
                        st.balloons()
                        # Clear cache to reload new model
                        get_model.clear()
                    else:
                        st.error(msg)
        else:
            st.warning("Cần thu thập ít nhất 100 frames dữ liệu trước khi huấn luyện.")

# ==========================================
# PAGE 2: DEMO THỰC TẾ
# ==========================================
elif page == "🏋️ 2. Triển khai mô hình (Demo)":
    st.markdown('<p class="main-header">Triển khai mô hình (Demo Thực Tế)</p>', unsafe_allow_html=True)
    
    st.write("Tải lên video tập luyện hoàn chỉnh để hệ thống phân tích, đếm số rep và đưa ra nhận xét tư thế.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        exercise_type = st.selectbox("Chọn bài tập", ["Push-up", "Squat"], key="demo_ex")
        uploaded_file = st.file_uploader("Chọn video demo", type=['mp4', 'avi', 'mov'], key="demo_vid")
        
    with col2:
        if uploaded_file is not None:
            if st.button("🚀 Bắt đầu phân tích", type="primary"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.write("Đang xử lý video...")
                progress_bar = st.progress(0)
                
                # Metrics UI
                m1, m2, m3 = st.columns(3)
                rep_metric = m1.empty()
                state_metric = m2.empty()
                conf_metric = m3.empty()
                
                frame_window = st.empty()
                
                processor = PoseProcessor()
                
                reps = 0
                current_state = "UP"
                frame_count = 0
                
                window_size = 30
                angle_history = []
                plot_data = {'frame': [], 'angle': []}
                
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    landmarks = processor.extract_keypoints(frame)
                    angles = processor.get_exercise_angles(landmarks)
                    
                    state = "UNKNOWN"
                    feedback = ""
                    confidence = 0.0
                    
                    if angles:
                        main_angle = angles['angle_elbow'] if exercise_type == "Push-up" else angles['angle_knee']
                        plot_data['frame'].append(frame_count)
                        plot_data['angle'].append(main_angle)
                        
                        if model_trained:
                            norm_angles = [
                                angles['angle_elbow']/180.0, 
                                angles['angle_hip']/180.0, 
                                angles['angle_knee']/180.0
                            ]
                            angle_history.append(norm_angles)
                            
                            if len(angle_history) > window_size:
                                angle_history.pop(0)
                                
                            if len(angle_history) == window_size:
                                window_arr = np.array(angle_history)
                                pred_phase, confidence = predict_phase(model, scaler, le, window_arr)
                                
                                if "down" in pred_phase:
                                    state = "DOWN"
                                elif "up" in pred_phase:
                                    state = "UP"
                        else:
                            state = processor.classify_state(angles, exercise_type)
                            confidence = 1.0
                            
                        if state == "DOWN" and current_state == "UP":
                            current_state = "DOWN"
                        elif state == "UP" and current_state == "DOWN":
                            current_state = "UP"
                            reps += 1
                            
                        feedback = processor.get_feedback(angles, exercise_type, current_state)
                        frame = processor.draw_overlay(frame, landmarks, state, reps, exercise_type, feedback)
                        
                    # Update Metrics UI
                    rep_metric.metric("Số Reps", reps)
                    state_metric.metric("Trạng thái", state)
                    conf_metric.metric("Độ tin cậy (Confidence)", f"{confidence*100:.1f}%")
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
                cap.release()
                exec_time = time.time() - start_time
                
                save_workout_history(exercise_type, reps, exec_time)
                
                st.success("✅ Xử lý hoàn tất! Kết quả đã được lưu vào Lịch sử tập luyện.")
                
                if plot_data['frame']:
                    fig = px.line(x=plot_data['frame'], y=plot_data['angle'], 
                                 labels={'x': 'Frame', 'y': 'Góc (độ)'},
                                 title="Biểu đồ góc theo thời gian")
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 3: KIỂM THỬ & ĐÁNH GIÁ
# ==========================================
elif page == "📈 3. Đánh giá & Hiệu năng":
    st.markdown('<p class="main-header">Đánh giá & Hiệu năng (Evaluation)</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Chỉ số & Biểu đồ kỹ thuật", "🎯 Kiểm thử Video Thực Tế"])
    
    with tab1:
        metrics = load_metrics()
        if not metrics:
            st.info("Chưa có dữ liệu đánh giá. Vui lòng huấn luyện mô hình trước.")
        else:
            st.write("### 1. Các chỉ số đo lường (Metrics)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy (Tập Test)", f"{metrics.get('accuracy', 0)*100:.2f}%")
            c2.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
            c3.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
            
            st.write("### 2. Biểu đồ Kỹ thuật")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("**Confusion Matrix**")
                cm = np.array(metrics.get('confusion_matrix', []))
                classes = metrics.get('classes', [])
                
                if len(cm) > 0 and len(classes) > 0:
                    fig_cm = px.imshow(cm, x=classes, y=classes, text_auto=True, 
                                      color_continuous_scale='Blues', aspect="auto")
                    fig_cm.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            with col_chart2:
                st.write("**Mức độ quan trọng của Đặc trưng (Feature Importance)**")
                if model_trained and hasattr(model, 'feature_importances_'):
                    # Random Forest feature importances
                    importances = model.feature_importances_
                    # We have window_size * 3 + 3 features. Just show top 10.
                    indices = np.argsort(importances)[::-1][:10]
                    top_importances = importances[indices]
                    top_features = [f"Feature {i}" for i in indices]
                    
                    fig_fi = px.bar(x=top_importances, y=top_features, orientation='h',
                                   labels={'x': 'Mức độ quan trọng', 'y': 'Đặc trưng'},
                                   title="Top 10 Đặc trưng quan trọng nhất")
                    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Không có thông tin Feature Importance.")
                    
            st.write("### 3. Phân tích sai số (Error Analysis)")
            st.markdown("""
            **Nhận định về các trường hợp dự đoán sai:**
            - **Nhầm lẫn ở pha Transition:** Mô hình thường gặp khó khăn tại ranh giới giữa pha `UP` và `DOWN`. Khi người tập dừng lại ở giữa chừng, góc khớp nằm ở mức lấp lửng, khiến độ tin cậy (Confidence) giảm mạnh.
            - **Sai lệch do góc quay:** Nếu camera đặt quá cao hoặc quá thấp, tỷ lệ cơ thể bị méo mó (perspective distortion), dẫn đến góc khớp do MediaPipe tính toán bị sai lệch so với thực tế.
            - **Tốc độ thực hiện:** Các rep thực hiện quá nhanh (motion blur) làm MediaPipe mất dấu keypoints, tạo ra các frame nhiễu.
            
            **Hướng cải thiện:**
            1. **Thu thập thêm dữ liệu đa dạng:** Bổ sung video từ nhiều góc quay khác nhau (chính diện, chéo 45 độ, ngang 90 độ) và nhiều điều kiện ánh sáng.
            2. **Áp dụng bộ lọc Kalman (Kalman Filter):** Làm mượt tọa độ keypoints trước khi đưa vào tính góc để giảm thiểu nhiễu do motion blur.
            3. **Chuyển đổi sang mô hình LSTM/GRU:** Thay vì dùng Sliding Window kết hợp Random Forest, mạng RNN (như LSTM) có khả năng ghi nhớ ngữ cảnh dài hạn tốt hơn, giúp xử lý các pha dừng nghỉ giữa chừng hiệu quả hơn.
            """)

    with tab2:
        st.write("Đưa vào video test và số rep thực tế để kiểm tra độ chính xác của mô hình.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            test_ex_type = st.selectbox("Loại bài tập", ["Push-up", "Squat"], key="test_ex")
            actual_reps = st.number_input("Số Reps thực tế trong video", min_value=1, value=5)
            uploaded_test_vid = st.file_uploader("Chọn video test", type=['mp4', 'avi', 'mov'], key="test_vid")
            
        with col2:
            if uploaded_test_vid is not None and model_trained:
                if st.button("Bắt đầu Kiểm thử", type="primary"):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                    tfile.write(uploaded_test_vid.read())
                    
                    st.write("Đang chạy kiểm thử...")
                    
                    cap = cv2.VideoCapture(tfile.name)
                    processor = PoseProcessor()
                    
                    reps = 0
                    current_state = "UP"
                    angle_history = []
                    window_size = 30
                    
                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_count = 0
                    start_time = time.time()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_count += 1
                        landmarks = processor.extract_keypoints(frame)
                        angles = processor.get_exercise_angles(landmarks)
                        
                        if angles:
                            norm_angles = [
                                angles['angle_elbow']/180.0, 
                                angles['angle_hip']/180.0, 
                                angles['angle_knee']/180.0
                            ]
                            angle_history.append(norm_angles)
                            
                            if len(angle_history) > window_size:
                                angle_history.pop(0)
                                
                            if len(angle_history) == window_size:
                                window_arr = np.array(angle_history)
                                pred_phase, _ = predict_phase(model, scaler, le, window_arr)
                                
                                if "down" in pred_phase:
                                    state = "DOWN"
                                elif "up" in pred_phase:
                                    state = "UP"
                                else:
                                    state = "UNKNOWN"
                                    
                                if state == "DOWN" and current_state == "UP":
                                    current_state = "DOWN"
                                elif state == "UP" and current_state == "DOWN":
                                    current_state = "UP"
                                    reps += 1
                                    
                        if total_frames > 0:
                            progress_bar.progress(min(frame_count / total_frames, 1.0))
                            
                    cap.release()
                    exec_time = time.time() - start_time
                    
                    error = abs(reps - actual_reps)
                    accuracy = max(0, 100 - (error / actual_reps * 100)) if actual_reps > 0 else 0
                    
                    save_workout_history(test_ex_type, reps, exec_time, accuracy=f"{accuracy:.1f}%")
                    
                    st.success("✅ Kiểm thử hoàn tất!")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Số Reps Thực Tế", actual_reps)
                    m2.metric("Số Reps Dự Đoán", reps, delta=reps-actual_reps)
                    m3.metric("Độ Chính Xác", f"{accuracy:.1f}%")
                    
            elif not model_trained:
                st.warning("Vui lòng huấn luyện mô hình trước khi kiểm thử.")

# ==========================================
# PAGE 4: LỊCH SỬ TẬP LUYỆN
# ==========================================
elif page == "📅 4. Lịch sử Tập luyện":
    st.markdown('<p class="main-header">Lịch sử Tập luyện</p>', unsafe_allow_html=True)
    
    history_file = 'data/workout_history.csv'
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        
        if not history_df.empty:
            st.write("### 📊 Thống kê Tổng quan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số lần tập", len(history_df))
            c2.metric("Tổng số Reps", history_df['reps'].sum())
            c3.metric("Tổng thời gian (s)", round(history_df['duration_seconds'].sum(), 1))
            
            st.write("### 📈 Biểu đồ Số Reps theo Thời gian")
            fig = px.bar(history_df, x='date', y='reps', color='exercise', 
                         title="Số Reps qua các buổi tập",
                         labels={'date': 'Thời gian', 'reps': 'Số Reps', 'exercise': 'Bài tập'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### 📋 Chi tiết các buổi tập")
            st.dataframe(history_df.sort_values(by='date', ascending=False), use_container_width=True)
            
            if st.button("🗑️ Xóa Lịch sử", type="secondary"):
                os.remove(history_file)
                st.success("Đã xóa lịch sử tập luyện!")
                st.rerun()
        else:
            st.info("Chưa có dữ liệu lịch sử tập luyện.")
    else:
        st.info("Chưa có dữ liệu lịch sử tập luyện. Hãy thực hiện Demo hoặc Kiểm thử để lưu lại kết quả.")

# ==========================================
# PAGE 5: BÁO CÁO HỌC MÁY
# ==========================================
elif page == "📄 5. Báo cáo chi tiết":
    st.markdown('<p class="main-header">Báo Cáo Nghiên Cứu Học Máy</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Tự động đếm số lần tập hít đất và squat từ video người tập bằng MediaPipe Pose kết hợp học máy</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ---
    ### 1. XÁC LẬP BÀI TOÁN
    
    **Bối cảnh thực tế & Động lực nghiên cứu**  
    Sự gia tăng của xu hướng tập luyện tại nhà (home workout) đặt ra một thách thức lớn: thiếu sự giám sát của huấn luyện viên (PT). Người tập thường đối mặt với hai vấn đề cốt lõi: sai tư thế (dẫn đến chấn thương) và đếm sai số lần tập (ảnh hưởng đến tiến độ và động lực). 
    
    **Tại sao chọn Push-up và Squat?**  
    Nghiên cứu tập trung vào hai bài tập này vì chúng là đại diện tiêu biểu cho các nhóm cơ phần trên (upper body) và phần dưới (lower body). Hơn nữa, đây là các bài tập đa khớp (compound movements) có tính chu kỳ rõ rệt, rất phù hợp để làm cơ sở đánh giá các mô hình nhận diện chuỗi thời gian. Việc "đếm số lần" (counting) mang lại giá trị định lượng quan trọng hơn việc chỉ "nhận diện" (classification) hành động, vì nó trực tiếp phục vụ việc theo dõi tiến độ tập luyện.
    
    **Đặc thù và Thách thức của Dữ liệu**  
    Dữ liệu đầu vào là video từ webcam hoặc điện thoại di động. Đây là nguồn dữ liệu "in-the-wild" mang tính thách thức cao do:
    - **Nhiễu (Noise):** Ánh sáng yếu, background phức tạp.
    - **Occlusion (Che khuất):** Các bộ phận cơ thể tự che khuất nhau ở một số góc quay.
    - **Góc nhìn (Viewpoint):** Người dùng hiếm khi đặt camera ở góc chuẩn 90 độ.
    
    ---
    ### 2. TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG
    
    **Tại sao sử dụng MediaPipe Pose thay vì Raw Video?**  
    Việc đưa trực tiếp chuỗi ảnh (raw frames) vào các mô hình 3D-CNN đòi hỏi tài nguyên tính toán khổng lồ và rất dễ bị overfit vào bối cảnh (background). MediaPipe Pose đóng vai trò như một bộ lọc thông tin, chuyển đổi không gian ảnh RGB nhiều chiều thành một vector 33 điểm neo (keypoints) 3D. Điều này giúp:
    1. Giảm triệt để chiều dữ liệu.
    2. Tăng tính bất biến (invariance) với ánh sáng, màu da, và quần áo.
    
    **Chiến lược Chuẩn hóa (Normalization)**  
    Tọa độ pixel thô không có ý nghĩa học máy nếu người tập đứng gần hoặc xa camera. Do đó, dữ liệu được chuẩn hóa bằng cách:
    - **Centering:** Tịnh tiến gốc tọa độ về điểm giữa hông (mid-hip), loại bỏ sự phụ thuộc vào vị trí camera.
    - **Scaling:** Chia tọa độ cho chiều cao thân người (khoảng cách vai - hông), giúp mô hình hoạt động ổn định bất kể chiều cao thực tế của người dùng.
    
    **Lựa chọn Đặc trưng (Feature Engineering)**  
    Thay vì dùng trực tiếp tọa độ (x, y, z), nghiên cứu trích xuất **Góc khớp (Joint Angles)** (ví dụ: góc khuỷu tay, góc đầu gối).  
    *Tại sao?* Vì góc khớp là đại lượng bất biến với phép quay và phép tịnh tiến của camera. Nó phản ánh trực tiếp cơ sinh học (biomechanics) của chuyển động. Một chuỗi các góc khớp theo thời gian (time-series) sẽ mô tả trọn vẹn một chu kỳ Lên-Xuống của bài tập.
    
    **Vai trò của Lọc nhiễu (Smoothing)**  
    MediaPipe thường gặp hiện tượng "jitter" (rung lắc điểm neo) ở các frame mờ. Nếu không có bộ lọc (như Moving Average hay Kalman Filter), các đỉnh nhiễu này sẽ đánh lừa logic đếm, tạo ra các "false reps" (đếm khống).
    
    ---
    ### 3. THỰC THI MÔ HÌNH
    
    **Phân tích các hướng tiếp cận:**
    - **(A) Rule-based (Dựa trên ngưỡng góc):** Dễ triển khai (VD: góc gối < 90° là xuống). *Nhược điểm:* Quá cứng nhắc. Mỗi người có biên độ khớp và tỷ lệ cơ thể khác nhau, một ngưỡng cố định sẽ thất bại trên diện rộng.
    - **(B) Machine Learning (Random Forest / SVM):** Rất phù hợp với dữ liệu dạng bảng (tabular data) như các góc khớp. Random Forest có khả năng xử lý tốt các mối quan hệ phi tuyến tính và ít bị overfit trên tập dữ liệu nhỏ.
    - **(C) Deep Learning (LSTM / GRU):** Là lựa chọn tối ưu về mặt lý thuyết vì bản chất bài tập là chuỗi thời gian. *Tuy nhiên*, LSTM đòi hỏi lượng dữ liệu lớn để hội tụ.
    
    **Quyết định thiết kế (Design Choice):**  
    Hệ thống hiện tại sử dụng **Random Forest** kết hợp với phương pháp **Sliding Window** (Cửa sổ trượt). Bằng cách đưa một chuỗi $N$ frames liên tiếp vào Random Forest, mô hình có thể nắm bắt được bối cảnh thời gian (temporal context) mà không cần kiến trúc phức tạp như LSTM, cân bằng hoàn hảo giữa độ chính xác và tốc độ suy luận (inference speed) trên CPU.
    
    **Tầm quan trọng của Online Learning / Lưu dữ liệu:**  
    Hệ thống cho phép người dùng tự thu thập video của chính mình để huấn luyện lại mô hình. Điều này tạo ra một hệ thống *Adaptive* (thích nghi), giúp mô hình tinh chỉnh (fine-tune) theo đúng form tập và góc quay quen thuộc của từng cá nhân, giải quyết triệt để bài toán "out-of-distribution".
    
    ---
    ### 4. PHÂN TÍCH & ĐÁNH GIÁ
    
    **Lựa chọn Metrics đánh giá:**
    - **Accuracy:** Đánh giá khả năng phân loại đúng trạng thái (Up/Down) trên từng frame.
    - **Precision/Recall:** Cực kỳ quan trọng để tránh đếm sai. Recall thấp nghĩa là bỏ sót rep (undercounting), Precision thấp nghĩa là đếm khống (overcounting).
    - **MAE (Mean Absolute Error):** Là metric thực tiễn nhất, đo lường sai lệch giữa số rep máy đếm và số rep thực tế do con người gán nhãn.
    
    **Nhận xét từ Confusion Matrix & Loss:**  
    Confusion Matrix giúp phát hiện các điểm mù của mô hình. Thường mô hình sẽ nhầm lẫn ở pha "Transition" (chuyển giao giữa Lên và Xuống). Việc trực quan hóa Prediction vs Ground Truth trên biểu đồ sóng (wave plot) cho thấy rõ: mô hình hoạt động cực tốt khi người dùng tập đúng nhịp, nhưng sẽ gặp khó khăn nếu người dùng dừng lại nghỉ quá lâu ở giữa một rep.
    
    **Sự đánh đổi (Trade-offs):**  
    Hệ thống chấp nhận hy sinh một phần nhỏ Accuracy (bằng cách dùng mô hình nhẹ như Random Forest thay vì các mạng Transformer nặng nề) để đổi lấy **Real-time Speed** (tốc độ xử lý thời gian thực) – yếu tố sống còn của một ứng dụng Fitness.
    
    ---
    ### 5. RỦI RO, ĐẠO ĐỨC & HƯỚNG MỞ RỘNG
    
    **Phân tích rủi ro (Risk Analysis):**  
    Điểm yếu chí mạng của hệ thống là sự phụ thuộc hoàn toàn vào MediaPipe. Nếu MediaPipe thất bại trong việc nhận diện bộ xương (do ánh sáng quá tối hoặc mặc đồ quá rộng), toàn bộ pipeline phía sau sẽ sụp đổ (Cascading Failure). Ngoài ra, với lượng dữ liệu nhỏ, mô hình dễ bị overfitting vào góc quay cụ thể.
    
    **So sánh với các giải pháp khác:**  
    So với việc dùng cảm biến đeo tay (wearable sensors) hay camera hồng ngoại (Kinect), giải pháp Vision-based qua webcam có độ chính xác thấp hơn một chút nhưng lại vượt trội hoàn toàn về tính tiện lợi, chi phí bằng 0 và khả năng tiếp cận đại chúng.
    
    **Đạo đức & Quyền riêng tư (Ethics & Privacy):**  
    Video tập luyện chứa hình ảnh nhạy cảm và không gian riêng tư của người dùng. Giải pháp thiết kế ở đây là **Edge Computing**: toàn bộ quá trình trích xuất MediaPipe và suy luận ML được thực hiện cục bộ (local) trên thiết bị. Không có raw video nào được gửi lên server, đảm bảo tuyệt đối quyền riêng tư.
    
    **Định hướng tương lai (Future Work):**
    1. **Mở rộng bài tập:** Kiến trúc hiện tại hoàn toàn có thể scale lên các bài tập khác như Plank, Pull-up, Jumping Jacks chỉ bằng cách thay đổi tập dữ liệu huấn luyện.
    2. **AI Coach (Huấn luyện viên AI):** Chuyển từ "Passive Counting" (chỉ đếm) sang "Active Correction" (chủ động sửa sai). Bằng cách phân tích quỹ đạo góc khớp, hệ thống có thể phát ra cảnh báo âm thanh realtime: *"Bạn đang chụm đầu gối quá nhiều"* hoặc *"Hãy xuống sâu hơn"*.
    """)
