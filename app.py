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
    ["📊 Dữ liệu & Huấn luyện", "🏋️ Demo Thực Tế", "📈 Kiểm thử & Đánh giá", "📅 Lịch sử Tập luyện"]
)

st.sidebar.markdown("---")

model, scaler, le = get_model()
model_trained = model is not None

if not model_trained and page != "📊 Dữ liệu & Huấn luyện":
    st.warning("⚠️ Mô hình chưa được huấn luyện! Vui lòng thu thập dữ liệu và huấn luyện trước.")

# ==========================================
# PAGE 1: DỮ LIỆU & HUẤN LUYỆN
# ==========================================
if page == "📊 Dữ liệu & Huấn luyện":
    st.markdown('<p class="main-header">Quản lý Dữ liệu & Huấn luyện</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📥 Thu thập Dữ liệu từ Video", "🧠 Huấn luyện Mô hình"])
    
    with tab1:
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
                        
        st.markdown("---")
        st.write("### 📊 Thống kê Tập dữ liệu hiện tại")
        df = get_dataset()
        
        if df is not None and not df.empty:
            st.write(f"**Tổng số frames:** {len(df)}")
            
            c1, c2 = st.columns(2)
            with c1:
                counts = df['phase_label'].value_counts().reset_index()
                counts.columns = ['phase_label', 'count']
                fig = px.bar(counts, 
                             x='phase_label', y='count', color='phase_label',
                             title="Phân phối Nhãn (Labels)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("Tập dữ liệu đang trống. Vui lòng tải lên video để thu thập dữ liệu.")

    with tab2:
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
elif page == "🏋️ Demo Thực Tế":
    st.markdown('<p class="main-header">Demo Đếm Reps Thực Tế</p>', unsafe_allow_html=True)
    
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
                                pred_phase, _ = predict_phase(model, scaler, le, window_arr)
                                
                                if "down" in pred_phase:
                                    state = "DOWN"
                                elif "up" in pred_phase:
                                    state = "UP"
                        else:
                            state = processor.classify_state(angles, exercise_type)
                            
                        if state == "DOWN" and current_state == "UP":
                            current_state = "DOWN"
                        elif state == "UP" and current_state == "DOWN":
                            current_state = "UP"
                            reps += 1
                            
                        feedback = processor.get_feedback(angles, exercise_type, current_state)
                        frame = processor.draw_overlay(frame, landmarks, state, reps, exercise_type, feedback)
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
                cap.release()
                exec_time = time.time() - start_time
                
                save_workout_history(exercise_type, reps, exec_time)
                
                st.success("✅ Xử lý hoàn tất! Kết quả đã được lưu vào Lịch sử tập luyện.")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số Reps", reps)
                m2.metric("Thời gian xử lý", f"{exec_time:.1f}s")
                m3.metric("FPS trung bình", f"{frame_count/exec_time:.1f}")
                
                if plot_data['frame']:
                    fig = px.line(x=plot_data['frame'], y=plot_data['angle'], 
                                 labels={'x': 'Frame', 'y': 'Góc (độ)'},
                                 title="Biểu đồ góc theo thời gian")
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 3: KIỂM THỬ & ĐÁNH GIÁ
# ==========================================
elif page == "📈 Kiểm thử & Đánh giá":
    st.markdown('<p class="main-header">Kiểm thử & Đánh giá Mô hình</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Kiểm thử Video Thực Tế", "📊 Chỉ số Mô hình (Metrics)"])
    
    with tab1:
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

    with tab2:
        metrics = load_metrics()
        if not metrics:
            st.info("Chưa có dữ liệu đánh giá. Vui lòng huấn luyện mô hình trước.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy (Tập Test)", f"{metrics.get('accuracy', 0)*100:.2f}%")
            c2.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
            c3.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
            
            st.write("### Confusion Matrix")
            cm = np.array(metrics.get('confusion_matrix', []))
            classes = metrics.get('classes', [])
            
            if len(cm) > 0 and len(classes) > 0:
                fig_cm = px.imshow(cm, x=classes, y=classes, text_auto=True, 
                                  color_continuous_scale='Blues', aspect="auto")
                fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# PAGE 4: LỊCH SỬ TẬP LUYỆN
# ==========================================
elif page == "📅 Lịch sử Tập luyện":
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
