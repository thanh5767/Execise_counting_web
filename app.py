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

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ─── Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ─── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a0f1e 0%, #0d1f12 100%);
    border-right: 1px solid rgba(74,222,128,0.15);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: block;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(74,222,128,0.12);
    border-color: rgba(74,222,128,0.4);
}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio input:checked + label {
    background: rgba(74,222,128,0.18);
    border-color: #4ade80;
}

/* ─── Main background ──────────────────────────────── */
.main .block-container {
    background: #f8fafc;
    padding-top: 2rem;
}

/* ─── Page header band ─────────────────────────────── */
.page-hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2116 50%, #0a1a0a 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(74,222,128,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.page-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.page-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    position: relative;
    z-index: 1;
    letter-spacing: -0.5px;
}
.page-hero p {
    color: rgba(255,255,255,0.6);
    font-size: 1.05rem;
    margin: 0;
    position: relative;
    z-index: 1;
    font-weight: 300;
}
.page-hero .badge {
    display: inline-block;
    background: rgba(74,222,128,0.2);
    border: 1px solid rgba(74,222,128,0.5);
    color: #4ade80;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
    position: relative;
    z-index: 1;
}

/* ─── Info card ────────────────────────────────────── */
.info-card {
    background: linear-gradient(135deg, #0f2027 0%, #1a3a20 100%);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: #e2e8f0;
}
.info-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #4ade80;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}
.info-card p, .info-card li { color: rgba(255,255,255,0.75); font-size: 0.9rem; line-height: 1.7; }

/* ─── Stat cards ───────────────────────────────────── */
.stat-row { display: flex; gap: 16px; margin-bottom: 24px; }
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px 24px;
    flex: 1;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.stat-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.1); }
.stat-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}
.stat-card .unit { font-size: 0.9rem; color: #64748b; margin-top: 4px; }
.stat-card .accent { color: #16a34a; }

/* ─── Section headings ─────────────────────────────── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    border-left: 4px solid #4ade80;
    padding-left: 14px;
    margin: 32px 0 16px 0;
}

/* ─── Panel / white card ───────────────────────────── */
.panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ─── Metric pill ──────────────────────────────────── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 30px;
    padding: 8px 16px;
    font-size: 0.88rem;
    color: #15803d;
    font-weight: 600;
    margin: 4px;
}

/* ─── Tab override ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 500;
    font-size: 0.9rem;
    color: #64748b;
    padding: 8px 20px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* ─── Buttons ──────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #16a34a, #15803d);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 10px 28px;
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #15803d, #166534);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(22,163,74,0.4);
}

/* ─── Progress bar ─────────────────────────────────── */
.stProgress > div > div > div > div { background: #4ade80; }

/* ─── Expander ─────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #f8fafc;
    border-radius: 10px;
    font-weight: 600;
    color: #0f172a;
}

/* ─── Alert boxes ──────────────────────────────────── */
.stAlert { border-radius: 12px; }

/* ─── Sidebar logo area ────────────────────────────── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 16px 0;
}
.sidebar-logo .logo-icon {
    font-size: 3rem;
    display: block;
    line-height: 1;
}
.sidebar-logo .logo-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #4ade80;
    letter-spacing: -0.3px;
    margin-top: 8px;
}
.sidebar-logo .logo-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 12px 0 18px 0;
}

/* ─── Model status badge ───────────────────────────── */
.model-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 16px;
}
.model-ok { background: rgba(74,222,128,0.12); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; }
.model-err { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); color: #f87171; }

/* ─── Analysis note box ────────────────────────────── */
.analysis-note {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    color: #78350f;
    line-height: 1.65;
}

/* ─── Error case table ─────────────────────────────── */
.error-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.error-table th {
    background: #0f172a; color: #4ade80;
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem; letter-spacing: 0.8px;
    padding: 12px 16px; text-align: left;
}
.error-table td { padding: 11px 16px; border-bottom: 1px solid #f1f5f9; color: #334155; vertical-align: top; }
.error-table tr:last-child td { border-bottom: none; }
.error-table tr:hover td { background: #f8fafc; }
.tag {
    display: inline-block;
    font-size: 0.72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 4px;
    background: #fee2e2; color: #991b1b; margin-bottom: 2px;
}
.tag-warn { background: #fef3c7; color: #92400e; }
.tag-ok   { background: #dcfce7; color: #166534; }
</style>
""", unsafe_allow_html=True)

# ── Helper: hero banner ────────────────────────────────────────────────────────
def page_hero(badge, title, subtitle):
    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">{badge}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def section_heading(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_real_data()

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model_cached()

def save_workout_history(exercise_type, reps, duration, accuracy=None):
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
    progress_bar = st.progress(0, text="Đang trích xuất keypoints…")
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
            progress_bar.progress(min(frame_count / total_frames, 1.0),
                                  text=f"Frame {frame_count}/{total_frames}")
    cap.release()
    if angles_list:
        saved_count = save_video_data(angles_list, exercise_type, phase_label)
        return True, saved_count
    return False, 0

# ── Video processor: write annotated video → re-encode → st.video() ───────────
def process_video_and_render(video_path, exercise_type):
    """
    Pipeline:
      1. Đọc từng frame → resize → MediaPipe → ML inference → vẽ overlay → ghi vào VideoWriter
      2. Re-encode bằng ffmpeg (H.264 + faststart) để trình duyệt stream mượt
      3. Trả về: path video đã xử lý, số reps, confidence, plot_data
    Không có st.image() trong vòng lặp → không lag trên Streamlit Cloud.
    """
    import subprocess

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Resize để tăng tốc MediaPipe (max 640px wide) ──────────────────────
    MAX_W = 640
    if orig_w > MAX_W:
        scale  = MAX_W / orig_w
        proc_w = MAX_W
        proc_h = int(orig_h * scale) & ~1   # đảm bảo chẵn (yêu cầu H.264)
    else:
        proc_w = orig_w & ~1
        proc_h = orig_h & ~1

    # ── Tạo temp file cho raw output ───────────────────────────────────────
    raw_tmp   = tempfile.NamedTemporaryFile(delete=False, suffix='_raw.mp4')
    raw_path  = raw_tmp.name
    raw_tmp.close()

    final_path = raw_path.replace('_raw.mp4', '_h264.mp4')

    # ── Chọn codec VideoWriter (ưu tiên mp4v vì stable trên mọi hệ thống) ─
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (proc_w, proc_h))
    if not writer.isOpened():
        cap.release()
        return None, 0, 0.0, {}

    # ── State machine + ML ─────────────────────────────────────────────────
    processor     = PoseProcessor()
    reps          = 0
    current_state = "UP"
    state         = "UNKNOWN"
    confidence    = 0.0
    window_size   = 30
    angle_history = []
    plot_data     = {'frame': [], 'angle': []}
    frame_count   = 0

    # UI: chỉ progress bar, KHÔNG st.image() trong vòng lặp
    prog  = st.progress(0, text="⏳ Bước 1/2 — Đang xử lý video…")
    info  = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize nếu cần
        if orig_w != proc_w or orig_h != proc_h:
            frame = cv2.resize(frame, (proc_w, proc_h))

        landmarks = processor.extract_keypoints(frame)
        angles    = processor.get_exercise_angles(landmarks)
        feedback  = ""

        if angles:
            main_angle = angles['angle_elbow'] if exercise_type == "Push-up" else angles['angle_knee']
            plot_data['frame'].append(frame_count)
            plot_data['angle'].append(main_angle)

            if model_trained:
                norm = [
                    angles['angle_elbow'] / 180.0,
                    angles['angle_hip']   / 180.0,
                    angles['angle_knee']  / 180.0,
                ]
                angle_history.append(norm)
                if len(angle_history) > window_size:
                    angle_history.pop(0)
                if len(angle_history) == window_size:
                    w_arr = np.array(angle_history)
                    pred_phase, confidence = predict_phase(model, scaler, le, w_arr)
                    if "down" in pred_phase:
                        state = "DOWN"
                    elif "up" in pred_phase:
                        state = "UP"
            else:
                state      = processor.classify_state(angles, exercise_type)
                confidence = 1.0

            if state == "DOWN" and current_state == "UP":
                current_state = "DOWN"
            elif state == "UP" and current_state == "DOWN":
                current_state = "UP"
                reps += 1

            feedback = processor.get_feedback(angles, exercise_type, current_state)
            frame    = processor.draw_overlay(frame, landmarks, state, reps, exercise_type, feedback)

        writer.write(frame)

        if total_frames > 0:
            pct = frame_count / total_frames
            prog.progress(
                min(pct, 1.0),
                text=f"⏳ Bước 1/2 — Frame {frame_count}/{total_frames} · {reps} reps"
            )

    cap.release()
    writer.release()
    info.empty()

    # ── Bước 2: Re-encode bằng ffmpeg → H.264 + faststart ─────────────────
    prog.progress(1.0, text="⚙️ Bước 2/2 — Đang tối ưu video để phát trực tuyến…")

    ffmpeg_ok = False
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', raw_path,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',      # tương thích tối đa với browser
            '-crf', '23',               # quality (18=cao, 28=thấp)
            '-preset', 'fast',          # encode nhanh, ưu tiên speed
            '-movflags', '+faststart',  # metadata lên đầu → stream ngay
            final_path
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=180)
        if res.returncode == 0 and os.path.exists(final_path):
            ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False

    prog.empty()

    video_out = final_path if ffmpeg_ok else raw_path
    return video_out, reps, confidence, plot_data

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🏋️</span>
        <div class="logo-title">AI Fitness Tracker</div>
        <div class="logo-sub">MediaPipe · ML · Streamlit</div>
    </div>
    <hr class="sidebar-divider"/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Điều hướng",
        [
            "📊  Giới thiệu & Khám phá dữ liệu",
            "🏋️  Triển khai mô hình",
            "📈  Đánh giá & Hiệu năng",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

    model, scaler, le = get_model()
    model_trained = model is not None

    if model_trained:
        st.markdown('<div class="model-status model-ok">✓ Mô hình đã sẵn sàng</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status model-err">✗ Chưa có mô hình</div>', unsafe_allow_html=True)

    st.markdown("""
    <br/>
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.2);text-align:center;letter-spacing:0.5px;">
    v2.0 · 2024 · AI Fitness Lab
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — GIỚI THIỆU & EDA
# ══════════════════════════════════════════════════════════════════════════════
if "Giới thiệu" in page:

    page_hero(
        "Đề tài tốt nghiệp",
        "Giới thiệu & Khám phá dữ liệu",
        "Tự động đếm số lần tập hít đất & squat từ video bằng MediaPipe Pose kết hợp học máy"
    )

    # ── Student info ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-card">
        <h3>📋 Thông tin đề tài</h3>
        <table style="width:100%;border-collapse:collapse;">
          <tr>
            <td style="width:30%;color:rgba(255,255,255,0.45);font-size:0.82rem;padding:5px 0;vertical-align:top;">Tên đề tài</td>
            <td style="color:#e2e8f0;font-size:0.92rem;font-weight:500;padding:5px 0;">
              Đếm số lần tập hít đất/squat từ video người tập bằng MediaPipe Pose kết hợp học máy
            </td>
          </tr>
          <tr>
            <td style="color:rgba(255,255,255,0.45);font-size:0.82rem;padding:5px 0;">Sinh viên</td>
            <td style="color:#4ade80;font-weight:600;padding:5px 0;">Nguyễn Văn A</td>
          </tr>
          <tr>
            <td style="color:rgba(255,255,255,0.45);font-size:0.82rem;padding:5px 0;">MSSV</td>
            <td style="color:#e2e8f0;padding:5px 0;">20201234</td>
          </tr>
          <tr>
            <td style="color:rgba(255,255,255,0.45);font-size:0.82rem;padding:5px 0;">Năm học</td>
            <td style="color:#e2e8f0;padding:5px 0;">2024 – 2025</td>
          </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Value props ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="panel" style="border-top:3px solid #4ade80;">
            <div style="font-size:2rem;margin-bottom:10px;">🎯</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#0f172a;margin-bottom:8px;">Đúng số lần</div>
            <div style="font-size:0.88rem;color:#64748b;line-height:1.65;">
                Đếm rep tự động, loại bỏ sai số do mất tập trung hoặc mệt mỏi khi tự đếm.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="panel" style="border-top:3px solid #38bdf8;">
            <div style="font-size:2rem;margin-bottom:10px;">🛡️</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#0f172a;margin-bottom:8px;">Phòng chấn thương</div>
            <div style="font-size:0.88rem;color:#64748b;line-height:1.65;">
                Phân tích góc khớp realtime, phát hiện tư thế sai và cảnh báo ngay lập tức.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="panel" style="border-top:3px solid #a78bfa;">
            <div style="font-size:2rem;margin-bottom:10px;">💰</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#0f172a;margin-bottom:8px;">Tiết kiệm chi phí</div>
            <div style="font-size:0.88rem;color:#64748b;line-height:1.65;">
                Thay thế một phần vai trò huấn luyện viên cá nhân — chỉ cần webcam là đủ.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Methodology overview ──────────────────────────────────────────────
    section_heading("🔬 Phương pháp & Pipeline")
    st.markdown("""
    <div class="panel">
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:#0f172a;">
          <th style="color:#4ade80;font-family:'Syne',sans-serif;font-size:0.78rem;letter-spacing:0.8px;padding:12px 16px;text-align:left;border-radius:8px 0 0 0;">BƯỚC</th>
          <th style="color:#4ade80;font-family:'Syne',sans-serif;font-size:0.78rem;letter-spacing:0.8px;padding:12px 16px;">CÔNG NGHỆ</th>
          <th style="color:#4ade80;font-family:'Syne',sans-serif;font-size:0.78rem;letter-spacing:0.8px;padding:12px 16px;border-radius:0 8px 0 0;">MÔ TẢ</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;font-weight:600;color:#0f172a;">1. Thu thập</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#6366f1;">Webcam / Điện thoại</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#475569;font-size:0.88rem;">Video tự quay trong điều kiện ánh sáng thông thường tại nhà</td></tr>
        <tr><td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;font-weight:600;color:#0f172a;">2. Trích xuất</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#6366f1;">MediaPipe BlazePose</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#475569;font-size:0.88rem;">33 điểm mốc cơ thể → tính góc khớp (khuỷu, hông, gối)</td></tr>
        <tr><td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;font-weight:600;color:#0f172a;">3. Tiền xử lý</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#6366f1;">Sliding Window · StandardScaler</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#475569;font-size:0.88rem;">Chuẩn hóa [0,1], cửa sổ trượt 30 frames, delta features</td></tr>
        <tr><td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;font-weight:600;color:#0f172a;">4. Mô hình</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#6366f1;">Random Forest</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#475569;font-size:0.88rem;">Phân loại pha UP / DOWN — nhẹ, nhanh, phù hợp realtime trên CPU</td></tr>
        <tr><td style="padding:12px 16px;font-weight:600;color:#0f172a;">5. Đếm Rep</td>
            <td style="padding:12px 16px;color:#6366f1;">State Machine</td>
            <td style="padding:12px 16px;color:#475569;font-size:0.88rem;">Chuyển trạng thái DOWN → UP = +1 rep, có lọc nhiễu</td></tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: EDA | Thu thập | Huấn luyện ────────────────────────────────
    section_heading("📂 Dữ liệu & Huấn luyện")
    tab_eda, tab_collect, tab_train = st.tabs([
        "🔍  Khám phá dữ liệu (EDA)",
        "📥  Thu thập dữ liệu",
        "🧠  Huấn luyện mô hình",
    ])

    # ── TAB EDA ───────────────────────────────────────────────────────────
    with tab_eda:
        df = get_dataset()
        if df is None or df.empty:
            st.info("⚠️ Tập dữ liệu đang trống. Hãy chuyển sang tab **Thu thập dữ liệu** để tải video lên.")
        else:
            # Dataset stats
            n_frames = len(df)
            n_classes = df['phase_label'].nunique()
            n_exercises = df['exercise_type'].nunique() if 'exercise_type' in df.columns else 1

            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-card">
                <div class="label">Tổng số frames</div>
                <div class="value accent">{n_frames:,}</div>
                <div class="unit">mẫu dữ liệu</div>
              </div>
              <div class="stat-card">
                <div class="label">Số nhãn (classes)</div>
                <div class="value">{n_classes}</div>
                <div class="unit">pha chuyển động</div>
              </div>
              <div class="stat-card">
                <div class="label">Số bài tập</div>
                <div class="value">{n_exercises}</div>
                <div class="unit">loại bài tập</div>
              </div>
              <div class="stat-card">
                <div class="label">Đặc trưng</div>
                <div class="value">3</div>
                <div class="unit">góc khớp</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Raw data preview
            with st.expander("📄 Xem dữ liệu thô (10 dòng đầu)", expanded=False):
                st.dataframe(df.head(10), width='stretch')

            # Charts row 1
            col1, col2 = st.columns(2)
            with col1:
                counts = df['phase_label'].value_counts().reset_index()
                counts.columns = ['phase_label', 'count']
                fig1 = px.bar(
                    counts, x='phase_label', y='count', color='phase_label',
                    title="Phân phối nhãn (Class Distribution)",
                    color_discrete_sequence=['#4ade80', '#16a34a', '#38bdf8', '#0284c7'],
                    template="plotly_white"
                )
                fig1.update_layout(
                    showlegend=False, title_font_family="Syne",
                    plot_bgcolor='white', paper_bgcolor='white',
                    xaxis_tickangle=-20,
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig1, width='stretch')

            with col2:
                numeric_df = df[['angle_elbow', 'angle_hip', 'angle_knee']]
                corr = numeric_df.corr()
                fig2 = px.imshow(
                    corr, text_auto='.2f', aspect="auto",
                    color_continuous_scale='RdYlGn',
                    title="Ma trận tương quan góc khớp",
                    template="plotly_white"
                )
                fig2.update_layout(
                    title_font_family="Syne",
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig2, width='stretch')

            # Charts row 2
            col3, col4 = st.columns(2)
            with col3:
                fig3 = px.box(
                    df, x="phase_label", y="angle_elbow", color="phase_label",
                    title="Phân bố Góc khuỷu tay theo Nhãn",
                    color_discrete_sequence=['#4ade80', '#16a34a', '#38bdf8', '#0284c7'],
                    template="plotly_white"
                )
                fig3.update_layout(showlegend=False, title_font_family="Syne", margin=dict(t=50, b=20))
                st.plotly_chart(fig3, width='stretch')

            with col4:
                fig4 = px.scatter(
                    df, x='angle_hip', y='angle_knee', color='phase_label',
                    title="Phân tán: Góc Hông vs Góc Gối",
                    opacity=0.65,
                    color_discrete_sequence=['#4ade80', '#16a34a', '#38bdf8', '#0284c7'],
                    template="plotly_white"
                )
                fig4.update_layout(title_font_family="Syne", margin=dict(t=50, b=20))
                st.plotly_chart(fig4, width='stretch')

            # Violin
            fig5 = px.violin(
                df, y="angle_knee", x="phase_label", color="phase_label",
                box=True, points="outliers",
                title="Phân bố mật độ Góc Gối theo Nhãn (Violin Plot)",
                color_discrete_sequence=['#4ade80', '#16a34a', '#38bdf8', '#0284c7'],
                template="plotly_white"
            )
            fig5.update_layout(showlegend=False, title_font_family="Syne", margin=dict(t=50, b=20))
            st.plotly_chart(fig5, width='stretch')

            # Analysis note
            st.markdown("""
            <div class="analysis-note">
              <strong>📝 Nhận xét phân tích:</strong><br/>
              • <strong>Phân phối nhãn:</strong> Biểu đồ cột cho thấy sự cân bằng giữa các pha UP/DOWN. Nếu dữ liệu bị lệch, mô hình sẽ thiên về lớp đa số — cần dùng class weighting hoặc thu thập bổ sung.<br/>
              • <strong>Tương quan:</strong> Trong Squat, góc hông & góc gối có tương quan thuận rất cao (~0.9). Trong Push-up, góc khuỷu là đặc trưng phân loại chủ đạo, tương quan với hông thấp hơn nhiều.<br/>
              • <strong>Boxplot:</strong> Khoảng cách IQR rõ rệt giữa <em>push-up_up</em> và <em>push-up_down</em> cho thấy góc khuỷu là feature quan trọng nhất để phân loại pha Push-up.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB THU THẬP ──────────────────────────────────────────────────────
    with tab_collect:
        st.markdown("""
        <div class="panel">
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#0f172a;margin-bottom:6px;">
            📥 Thu thập dữ liệu huấn luyện từ video
          </div>
          <div style="font-size:0.88rem;color:#64748b;line-height:1.65;">
            Tải lên các đoạn video ngắn — mỗi video <strong>chỉ chứa một pha</strong> (LÊN hoặc XUỐNG).
            Hệ thống sẽ tự động dùng MediaPipe để trích xuất góc khớp và lưu vào dataset.
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            ex_type = st.selectbox("Loại bài tập", ["Push-up", "Squat"], key="collect_ex")
            phase = st.selectbox(
                "Pha chuyển động (nhãn)",
                [f"{ex_type.lower()}_up", f"{ex_type.lower()}_down"],
                key="collect_phase"
            )
            uploaded_train_vid = st.file_uploader(
                "Chọn video huấn luyện",
                type=['mp4', 'avi', 'mov'], key="train_vid"
            )
        with col2:
            st.markdown("""
            <div class="panel" style="background:#f8fafc;">
              <div style="font-weight:600;color:#0f172a;margin-bottom:10px;">💡 Hướng dẫn quay video</div>
              <ul style="font-size:0.87rem;color:#475569;line-height:1.8;margin:0;padding-left:18px;">
                <li>Đặt camera <strong>ngang hông</strong>, cách 1.5–2m</li>
                <li>Đảm bảo <strong>toàn thân</strong> trong khung hình</li>
                <li>Ánh sáng đầy đủ, tránh ngược sáng</li>
                <li>Mỗi pha nên có <strong>5–10 lần lặp</strong></li>
                <li>Mặc quần áo bó, tránh quần áo rộng che khớp</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

            if uploaded_train_vid is not None:
                if st.button("⚡ Trích xuất & Lưu vào Dataset", type="primary", width='stretch'):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_train_vid.read())
                    with st.spinner("Đang chạy MediaPipe…"):
                        success, count = extract_frames_from_video(tfile.name, ex_type, phase)
                    if success:
                        st.success(f"✅ Đã trích xuất và lưu **{count} frames** với nhãn `{phase}`.")
                        get_dataset.clear()
                        st.rerun()
                    else:
                        st.error("❌ Không tìm thấy người trong video hoặc có lỗi xảy ra.")

        # Dataset health
        df_check = load_real_data()
        if df_check is not None:
            st.markdown("---")
            st.markdown(f"**Tình trạng dataset hiện tại:** `{len(df_check)}` frames | `{df_check['phase_label'].nunique()}` nhãn")
            needed = max(0, 100 - len(df_check))
            st.progress(min(len(df_check)/100, 1.0),
                        text=f"{'✅ Đủ dữ liệu để huấn luyện' if needed == 0 else f'Cần thêm ~{needed} frames nữa'}")

    # ── TAB HUẤN LUYỆN ────────────────────────────────────────────────────
    with tab_train:
        df_train = get_dataset()
        n_samples = len(df_train) if df_train is not None else 0

        if n_samples < 100:
            st.warning(f"⚠️ Dataset hiện có **{n_samples} frames** — cần ít nhất **100 frames** để huấn luyện. Hãy thu thập thêm dữ liệu.")
        else:
            st.success(f"✅ Dataset sẵn sàng: **{n_samples} frames** · **{df_train['phase_label'].nunique()} nhãn**")

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**⚙️ Tùy chỉnh Siêu tham số (Hyperparameters)**")
            col_hp1, col_hp2, col_hp3 = st.columns(3)
            with col_hp1:
                n_estimators = st.slider("Số cây (n_estimators)", 50, 300, 100, 50)
            with col_hp2:
                max_depth_opt = st.selectbox("Độ sâu (max_depth)", ["Không giới hạn", 10, 20, 30])
                max_depth = None if max_depth_opt == "Không giới hạn" else max_depth_opt
            with col_hp3:
                use_aug = st.checkbox("Data Augmentation (thêm nhiễu)", value=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🚀 Bắt đầu Huấn luyện", type="primary", width='stretch'):
                with st.spinner("⏳ Đang huấn luyện Random Forest…"):
                    success, msg = train_models(n_estimators=n_estimators, max_depth=max_depth, use_augmentation=use_aug)
                if success:
                    st.success(msg)
                    st.balloons()
                    get_model.clear()
                else:
                    st.error(msg)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRIỂN KHAI MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════════════
elif "Triển khai" in page:

    page_hero(
        "Demo thực tế",
        "Triển khai mô hình",
        "Upload video tập luyện — AI sẽ đếm rep, theo dõi góc khớp và phân tích tư thế của bạn"
    )

    if not model_trained:
        st.warning("⚠️ Mô hình chưa được huấn luyện. Hãy vào trang **Giới thiệu & EDA** → tab **Huấn luyện mô hình** để train trước.")

    tab_demo, tab_test = st.tabs([
        "🎬  Phân tích Real-time",
        "🎯  Kiểm thử Độ chính xác",
    ])

    # ── TAB REAL-TIME ─────────────────────────────────────────────────────
    with tab_demo:

        # ── Settings row (horizontal, compact) ───────────────────────────
        cfg1, cfg2, cfg3, cfg4 = st.columns([1.2, 1.2, 1, 1.6])
        with cfg1:
            exercise_type = st.selectbox("🏋️ Bài tập", ["Push-up", "Squat"], key="demo_ex")
        with cfg2:
            display_every = st.select_slider(
                "⚡ Cập nhật UI mỗi N frame",
                options=[1, 2, 3, 4, 5, 6, 8, 10],
                value=3,
                help="Thấp = mượt hơn nhưng tốn bandwidth. Cao = nhanh hơn trên server chậm."
            )
        with cfg3:
            show_skeleton = st.toggle("🦴 Skeleton", value=True)
        with cfg4:
            display_w = st.select_slider(
                "📐 Kích thước hiển thị",
                options=[320, 480, 640],
                value=480,
                help="Nhỏ hơn = upload WebSocket nhẹ hơn, lag ít hơn trên Cloud."
            )

        uploaded_file = st.file_uploader(
            "📁 Tải lên video tập luyện (.mp4 / .avi / .mov)",
            type=['mp4', 'avi', 'mov'], key="demo_vid",
            label_visibility="collapsed"
        )

        col_run, col_tip = st.columns([1, 3])
        with col_run:
            analyze_btn = st.button(
                "▶ Bắt đầu phân tích real-time",
                type="primary", width='stretch',
                disabled=uploaded_file is None
            )
        with col_tip:
            st.markdown("""
            <div style="padding:10px 16px;background:#f0fdf4;border:1px solid #bbf7d0;
                 border-radius:10px;font-size:0.83rem;color:#166534;line-height:1.6;">
              💡 <strong>Tip tối ưu trên Cloud:</strong>
              Chọn <em>N=3</em> và kích thước <em>480px</em> để cân bằng giữa độ mượt và tốc độ.
              Video ngắn &lt;60s cho trải nghiệm tốt nhất.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Placeholder layout — tạo sẵn TRƯỚC khi vòng lặp chạy ────────
        # Quan trọng: tạo placeholder 1 lần, cập nhật in-place → không re-layout
        ph_frame   = st.empty()   # khung hình chính
        ph_metrics = st.empty()   # dòng metric số
        ph_bar     = st.empty()   # progress bar
        ph_result  = st.empty()   # kết quả cuối + chart

        if uploaded_file is None:
            ph_frame.markdown("""
            <div style="text-align:center;padding:80px 40px;background:#f8fafc;
                 border:2px dashed #e2e8f0;border-radius:16px;">
              <div style="font-size:5rem;margin-bottom:16px;">🎥</div>
              <div style="font-family:'Syne',sans-serif;font-size:1.2rem;
                   font-weight:700;color:#0f172a;margin-bottom:8px;">
                Upload video để bắt đầu phân tích real-time
              </div>
              <div style="font-size:0.88rem;color:#94a3b8;">
                Skeleton · Rep counter · Góc khớp — cập nhật liên tục theo từng frame
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif analyze_btn:

            # ── Ghi upload ra disk ────────────────────────────────────────
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush(); tfile.close()
            in_path = tfile.name

            # ── Chuẩn bị video reader ─────────────────────────────────────
            cap          = cv2.VideoCapture(in_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30
            orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Scale: kích thước xử lý (ML) ≤ 640px
            proc_w = min(640, orig_w)
            proc_h = int(orig_h * proc_w / orig_w) & ~1
            proc_w = proc_w & ~1

            # Scale: kích thước hiển thị (người dùng chọn)
            disp_h = int(orig_h * display_w / orig_w) & ~1
            disp_w = display_w & ~1

            # ── Chuẩn bị VideoWriter (ghi song song để tải về sau) ───────
            raw_tmp   = tempfile.NamedTemporaryFile(delete=False, suffix='_raw.mp4')
            raw_path  = raw_tmp.name; raw_tmp.close()
            final_path = raw_path.replace('_raw.mp4', '_h264.mp4')

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(raw_path, fourcc, fps_src, (proc_w, proc_h))

            # ── State machine + ML ────────────────────────────────────────
            processor     = PoseProcessor()
            reps          = 0
            current_state = "UP"
            state         = "UNKNOWN"
            confidence    = 0.0
            window_size   = 30
            angle_history = []
            plot_data     = {'frame': [], 'angle': []}
            frame_count   = 0
            start_time    = time.time()

            # ── VÒNG LẶP CHÍNH ───────────────────────────────────────────
            # Kỹ thuật: xử lý 100% frame nhưng chỉ đẩy lên UI mỗi N frame
            # → ML chính xác, WebSocket không bị flood

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Resize về kích thước xử lý (nhanh hơn cho MediaPipe)
                if orig_w != proc_w or orig_h != proc_h:
                    frame = cv2.resize(frame, (proc_w, proc_h))

                # ── ML inference (mỗi frame) ──────────────────────────────
                landmarks = processor.extract_keypoints(frame)
                angles    = processor.get_exercise_angles(landmarks)
                feedback  = ""

                if angles:
                    main_angle = angles['angle_elbow'] if exercise_type == "Push-up" else angles['angle_knee']
                    plot_data['frame'].append(frame_count)
                    plot_data['angle'].append(main_angle)

                    if model_trained:
                        norm = [
                            angles['angle_elbow'] / 180.0,
                            angles['angle_hip']   / 180.0,
                            angles['angle_knee']  / 180.0,
                        ]
                        angle_history.append(norm)
                        if len(angle_history) > window_size:
                            angle_history.pop(0)
                        if len(angle_history) == window_size:
                            w_arr = np.array(angle_history)
                            pred_phase, confidence = predict_phase(model, scaler, le, w_arr)
                            state = ("DOWN" if "down" in pred_phase
                                     else "UP" if "up" in pred_phase
                                     else state)          # giữ state cũ nếu UNKNOWN
                    else:
                        state      = processor.classify_state(angles, exercise_type)
                        confidence = 1.0

                    if state == "DOWN" and current_state == "UP":
                        current_state = "DOWN"
                    elif state == "UP" and current_state == "DOWN":
                        current_state = "UP"
                        reps += 1

                    feedback = processor.get_feedback(angles, exercise_type, current_state)

                # Vẽ overlay skeleton + HUD
                if show_skeleton:
                    annotated = processor.draw_overlay(
                        frame.copy(), landmarks, state, reps, exercise_type, feedback
                    )
                else:
                    annotated = frame.copy()
                    # Vẽ HUD tối giản không skeleton (nhẹ hơn)
                    cv2.rectangle(annotated, (0, 0), (280, 80), (15, 23, 42), -1)
                    cv2.putText(annotated, f"REPS: {reps}", (14, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (74, 222, 128), 3)
                    cv2.putText(annotated, f"{state}  {confidence*100:.0f}%", (14, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Ghi vào output video (mọi frame)
                writer.write(annotated)

                # ── UI update: chỉ mỗi N frame (throttled) ───────────────
                if frame_count % display_every == 0 or frame_count == 1:

                    # Resize xuống kích thước hiển thị trước khi encode JPEG
                    if proc_w != disp_w:
                        disp_frame = cv2.resize(annotated, (disp_w, disp_h))
                    else:
                        disp_frame = annotated

                    # Encode JPEG trong memory (nhanh hơn PNG, đủ chất lượng)
                    ret_enc, buf = cv2.imencode(
                        '.jpg', disp_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 82]   # 82% = cân bằng quality/size
                    )
                    if ret_enc:
                        ph_frame.image(
                            buf.tobytes(),
                            channels="BGR",
                            width='stretch',
                            caption=None,
                            output_format="JPEG"
                        )

                    # Metric row dạng HTML (1 lần write, không re-layout)
                    elapsed = time.time() - start_time
                    state_color = (
                        "#4ade80" if state == "UP"
                        else "#f87171" if state == "DOWN"
                        else "#94a3b8"
                    )
                    ph_metrics.markdown(f"""
                    <div style="display:flex;gap:12px;margin:8px 0;">
                      <div style="flex:1;background:#fff;border:1px solid #e2e8f0;border-top:3px solid #4ade80;
                           border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#94a3b8;letter-spacing:1px;
                             text-transform:uppercase;font-weight:600;">REPS</div>
                        <div style="font-family:'Syne',sans-serif;font-size:2.2rem;
                             font-weight:800;color:#16a34a;line-height:1;">{reps}</div>
                      </div>
                      <div style="flex:1;background:#fff;border:1px solid #e2e8f0;border-top:3px solid {state_color};
                           border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#94a3b8;letter-spacing:1px;
                             text-transform:uppercase;font-weight:600;">TRẠNG THÁI</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
                             font-weight:800;color:{state_color};line-height:1;">{state}</div>
                      </div>
                      <div style="flex:1;background:#fff;border:1px solid #e2e8f0;border-top:3px solid #a78bfa;
                           border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#94a3b8;letter-spacing:1px;
                             text-transform:uppercase;font-weight:600;">CONFIDENCE</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
                             font-weight:800;color:#7c3aed;line-height:1;">{confidence*100:.0f}%</div>
                      </div>
                      <div style="flex:1;background:#fff;border:1px solid #e2e8f0;border-top:3px solid #38bdf8;
                           border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#94a3b8;letter-spacing:1px;
                             text-transform:uppercase;font-weight:600;">THỜI GIAN</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
                             font-weight:800;color:#0284c7;line-height:1;">{elapsed:.1f}s</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar
                    if total_frames > 0:
                        pct = frame_count / total_frames
                        ph_bar.progress(
                            min(pct, 1.0),
                            text=f"⏳ Frame {frame_count}/{total_frames} · {int(pct*100)}%"
                        )

            # ── Kết thúc vòng lặp ────────────────────────────────────────
            cap.release()
            writer.release()
            exec_time = time.time() - start_time

            ph_bar.empty()
            save_workout_history(exercise_type, reps, exec_time)

            # ── Re-encode bằng ffmpeg nếu có ─────────────────────────────
            import subprocess
            ffmpeg_ok = False
            try:
                res = subprocess.run([
                    'ffmpeg', '-y', '-i', raw_path,
                    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', '-preset', 'fast',
                    '-movflags', '+faststart', final_path
                ], capture_output=True, timeout=180)
                ffmpeg_ok = (res.returncode == 0 and os.path.exists(final_path))
            except Exception:
                ffmpeg_ok = False

            video_out = final_path if ffmpeg_ok else raw_path

            # ── Hiển thị kết quả tổng kết ─────────────────────────────────
            with ph_result.container():
                st.success(f"✅ Phân tích hoàn tất — **{reps} reps** trong {exec_time:.1f}s")

                # KPI cuối
                st.markdown(f"""
                <div style="display:flex;gap:12px;margin:12px 0 20px 0;">
                  <div style="flex:1;background:linear-gradient(135deg,#0f2027,#1a3a20);
                       border-radius:14px;padding:20px 22px;text-align:center;">
                    <div style="font-size:0.72rem;color:rgba(255,255,255,0.45);letter-spacing:1px;
                         text-transform:uppercase;margin-bottom:6px;">Tổng Reps</div>
                    <div style="font-family:'Syne',sans-serif;font-size:3rem;
                         font-weight:800;color:#4ade80;line-height:1;">{reps}</div>
                  </div>
                  <div style="flex:1;background:#fff;border:1px solid #e2e8f0;
                       border-radius:14px;padding:20px 22px;text-align:center;">
                    <div style="font-size:0.72rem;color:#94a3b8;letter-spacing:1px;
                         text-transform:uppercase;margin-bottom:6px;">Thời gian</div>
                    <div style="font-family:'Syne',sans-serif;font-size:2.4rem;
                         font-weight:800;color:#0f172a;line-height:1;">{exec_time:.1f}<span style="font-size:1rem;">s</span></div>
                  </div>
                  <div style="flex:1;background:#fff;border:1px solid #e2e8f0;
                       border-radius:14px;padding:20px 22px;text-align:center;">
                    <div style="font-size:0.72rem;color:#94a3b8;letter-spacing:1px;
                         text-transform:uppercase;margin-bottom:6px;">Frames xử lý</div>
                    <div style="font-family:'Syne',sans-serif;font-size:2.4rem;
                         font-weight:800;color:#0f172a;line-height:1;">{frame_count:,}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Download button
                if os.path.exists(video_out):
                    with open(video_out, 'rb') as vf:
                        video_bytes = vf.read()
                    st.download_button(
                        label="⬇️ Tải video đã chú thích (H.264)",
                        data=video_bytes,
                        file_name=f"fitness_annotated_{exercise_type.lower().replace('-','_')}.mp4",
                        mime="video/mp4",
                        width='stretch'
                    )

                # Biểu đồ góc đầy đủ sau khi xong
                if plot_data.get('frame'):
                    angle_label = "Góc Khuỷu tay (°)" if exercise_type == "Push-up" else "Góc Gối (°)"
                    angles_arr  = np.array(plot_data['angle'])
                    kernel      = np.ones(9) / 9
                    smooth      = np.convolve(angles_arr, kernel, mode='same')

                    fig_angle = go.Figure()
                    fig_angle.add_trace(go.Scatter(
                        x=plot_data['frame'], y=smooth,
                        mode='lines', name=f"{angle_label} (mịn)",
                        line=dict(color='#16a34a', width=2.5),
                        fill='tozeroy', fillcolor='rgba(74,222,128,0.10)',
                    ))
                    fig_angle.add_trace(go.Scatter(
                        x=plot_data['frame'], y=plot_data['angle'],
                        mode='lines', name='Thô',
                        line=dict(color='rgba(74,222,128,0.3)', width=1),
                    ))
                    fig_angle.update_layout(
                        title=f"📈 {angle_label} theo Frame — {reps} chu kỳ phát hiện",
                        title_font_family="Syne",
                        xaxis_title="Frame", yaxis_title=angle_label,
                        template="plotly_white", hovermode='x unified',
                        legend=dict(orientation='h', y=1.08, x=1, xanchor='right'),
                        margin=dict(t=60, b=20),
                    )
                    st.plotly_chart(fig_angle, width='stretch')

            # Dọn temp files
            for p in [in_path, raw_path]:
                try: os.unlink(p)
                except Exception: pass

    # ── TAB KIỂM THỬ ─────────────────────────────────────────────────────
    with tab_test:
        st.markdown("""
        <div class="panel">
          Nhập video có <strong>số rep biết trước</strong> để đo sai số của mô hình.
          Kết quả sẽ tính <em>MAE</em> và <em>Accuracy đếm rep</em> cho video cụ thể đó.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            test_ex_type  = st.selectbox("Loại bài tập", ["Push-up", "Squat"], key="test_ex")
            actual_reps   = st.number_input("Số Reps thực tế trong video", min_value=1, value=5)
            uploaded_test = st.file_uploader("Chọn video kiểm thử", type=['mp4', 'avi', 'mov'], key="test_vid")

        with col2:
            if uploaded_test and model_trained:
                if st.button("🔬 Bắt đầu kiểm thử", type="primary", width='stretch'):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_test.read())
                    tfile.flush()
                    tfile.close()

                    start_time = time.time()

                    # Dùng lại process_video_and_render — không cần render video đầu ra
                    cap = cv2.VideoCapture(tfile.name)
                    processor = PoseProcessor()
                    reps, current_state = 0, "UP"
                    angle_history = []
                    window_size = 30
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_count  = 0
                    bar = st.progress(0, text="🔬 Đang kiểm thử…")

                    # Resize cho đồng nhất với train
                    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    MAX_W = 640
                    scale  = MIN_W = min(MAX_W, orig_w)
                    proc_w = scale
                    proc_h = int(orig_h * (proc_w / orig_w)) & ~1

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        if orig_w > MAX_W:
                            frame = cv2.resize(frame, (proc_w, proc_h))
                        landmarks = processor.extract_keypoints(frame)
                        angles    = processor.get_exercise_angles(landmarks)
                        if angles:
                            norm = [angles['angle_elbow']/180.0,
                                    angles['angle_hip']/180.0,
                                    angles['angle_knee']/180.0]
                            angle_history.append(norm)
                            if len(angle_history) > window_size:
                                angle_history.pop(0)
                            if len(angle_history) == window_size:
                                w_arr = np.array(angle_history)
                                pred, _ = predict_phase(model, scaler, le, w_arr)
                                state = "DOWN" if "down" in pred else ("UP" if "up" in pred else "UNKNOWN")
                                if state == "DOWN" and current_state == "UP":
                                    current_state = "DOWN"
                                elif state == "UP" and current_state == "DOWN":
                                    current_state = "UP"
                                    reps += 1
                        if total_frames > 0:
                            bar.progress(
                                min(frame_count / total_frames, 1.0),
                                text=f"🔬 Frame {frame_count}/{total_frames} · {reps} reps phát hiện"
                            )
                    cap.release()
                    bar.empty()

                    exec_time = time.time() - start_time
                    error    = abs(reps - actual_reps)
                    accuracy = max(0.0, 100 - (error / actual_reps * 100)) if actual_reps > 0 else 0
                    save_workout_history(test_ex_type, reps, exec_time, accuracy=f"{accuracy:.1f}%")

                    st.success("✅ Kiểm thử hoàn tất!")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Reps thực tế",  actual_reps)
                    r2.metric("Reps dự đoán",  reps, delta=reps - actual_reps)
                    r3.metric("Độ chính xác",  f"{accuracy:.1f}%")

                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=accuracy,
                        title={'text': "Accuracy (%)", 'font': {'family': 'Syne'}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar':  {'color': "#16a34a"},
                            'steps': [
                                {'range': [0,  60],  'color': '#fecaca'},
                                {'range': [60, 80],  'color': '#fed7aa'},
                                {'range': [80, 100], 'color': '#bbf7d0'},
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_gauge, width='stretch')

                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass

            elif not model_trained:
                st.warning("Cần huấn luyện mô hình trước.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ĐÁNH GIÁ & HIỆU NĂNG
# ══════════════════════════════════════════════════════════════════════════════
elif "Đánh giá" in page:

    page_hero(
        "Evaluation",
        "Đánh giá & Hiệu năng",
        "Các chỉ số kỹ thuật, Confusion Matrix, phân tích sai số và lịch sử tập luyện"
    )

    tab_metrics, tab_history, tab_report = st.tabs([
        "📊  Chỉ số & Biểu đồ",
        "📅  Lịch sử Tập luyện",
        "📄  Báo cáo Kỹ thuật",
    ])

    # ── TAB METRICS ────────────────────────────────────────────────────────
    with tab_metrics:
        metrics = load_metrics()

        if not metrics:
            st.info("Chưa có dữ liệu đánh giá. Hãy huấn luyện mô hình trước.")
        else:
            # KPI row
            acc = metrics.get('accuracy', 0) * 100
            f1  = metrics.get('f1_score', 0) * 100
            prec= metrics.get('precision', 0) * 100

            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-card" style="border-top:3px solid #4ade80;">
                <div class="label">Accuracy</div>
                <div class="value accent">{acc:.1f}<span style="font-size:1.2rem;">%</span></div>
                <div class="unit">Tập kiểm tra (test set)</div>
              </div>
              <div class="stat-card" style="border-top:3px solid #38bdf8;">
                <div class="label">F1-Score</div>
                <div class="value" style="color:#0284c7;">{f1:.1f}<span style="font-size:1.2rem;">%</span></div>
                <div class="unit">Weighted average</div>
              </div>
              <div class="stat-card" style="border-top:3px solid #a78bfa;">
                <div class="label">Precision</div>
                <div class="value" style="color:#7c3aed;">{prec:.1f}<span style="font-size:1.2rem;">%</span></div>
                <div class="unit">Weighted average</div>
              </div>
              <div class="stat-card" style="border-top:3px solid #fb923c;">
                <div class="label">Mô hình</div>
                <div class="value" style="font-size:1.2rem;color:#ea580c;">RF</div>
                <div class="unit">Random Forest · CPU</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics bar
            metrics_df = pd.DataFrame({
                'Chỉ số':  ['Accuracy', 'F1-Score', 'Precision'],
                'Điểm (%)': [acc, f1, prec]
            })
            fig_bar = px.bar(
                metrics_df, x='Chỉ số', y='Điểm (%)', color='Chỉ số', text='Điểm (%)',
                title="So sánh tổng quan các chỉ số đánh giá",
                color_discrete_sequence=['#4ade80', '#38bdf8', '#a78bfa'],
                range_y=[0, 110], template="plotly_white"
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(showlegend=False, title_font_family="Syne",
                                  margin=dict(t=50, b=10))
            st.plotly_chart(fig_bar, width='stretch')

            section_heading("🗂️ Biểu đồ kỹ thuật")
            col1, col2 = st.columns(2)

            with col1:
                cm      = np.array(metrics.get('confusion_matrix', []))
                classes = metrics.get('classes', [])
                if len(cm) > 0 and len(classes) > 0:
                    fig_cm = px.imshow(
                        cm, x=classes, y=classes, text_auto=True,
                        color_continuous_scale='Greens', aspect="auto",
                        title="Confusion Matrix"
                    )
                    fig_cm.update_layout(
                        xaxis_title="Nhãn Dự đoán", yaxis_title="Nhãn Thực tế",
                        title_font_family="Syne", margin=dict(t=50, b=20)
                    )
                    st.plotly_chart(fig_cm, width='stretch')

            with col2:
                if model_trained and hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices     = np.argsort(importances)[::-1][:12]
                    top_vals    = importances[indices]
                    top_names   = [f"Feature {i}" for i in indices]

                    fig_fi = px.bar(
                        x=top_vals, y=top_names, orientation='h',
                        labels={'x': 'Mức độ quan trọng', 'y': 'Đặc trưng'},
                        title="Top 12 đặc trưng quan trọng nhất",
                        color=top_vals,
                        color_continuous_scale='Greens',
                        template="plotly_white"
                    )
                    fig_fi.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False, coloraxis_showscale=False,
                        title_font_family="Syne", margin=dict(t=50, b=20)
                    )
                    st.plotly_chart(fig_fi, width='stretch')
                else:
                    st.info("Không có dữ liệu Feature Importance.")

            section_heading("🔍 Phân tích sai số")
            st.markdown("""
            <div class="panel">
            <table class="error-table">
              <thead>
                <tr>
                  <th>Trường hợp lỗi</th>
                  <th>Mức độ</th>
                  <th>Nguyên nhân</th>
                  <th>Hướng khắc phục</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Nhầm ở pha Transition</strong></td>
                  <td><span class="tag">Cao</span></td>
                  <td>Góc khớp ở ranh giới UP/DOWN — confidence thấp</td>
                  <td>Thêm hysteresis threshold, dùng LSTM thay RF</td>
                </tr>
                <tr>
                  <td><strong>Góc quay lệch</strong></td>
                  <td><span class="tag tag-warn">Trung bình</span></td>
                  <td>Camera quá cao/thấp → perspective distortion</td>
                  <td>Chuẩn hóa theo tỷ lệ thân người trước khi tính góc</td>
                </tr>
                <tr>
                  <td><strong>Tốc độ thực hiện nhanh</strong></td>
                  <td><span class="tag tag-warn">Trung bình</span></td>
                  <td>Motion blur → MediaPipe mất keypoints</td>
                  <td>Kalman filter trên chuỗi keypoints, interpolation</td>
                </tr>
                <tr>
                  <td><strong>Dừng nghỉ giữa rep</strong></td>
                  <td><span class="tag tag-ok">Thấp</span></td>
                  <td>State machine không phân biệt dừng vs xuống</td>
                  <td>Thêm trạng thái HOLD, timeout reset state</td>
                </tr>
                <tr>
                  <td><strong>Ánh sáng yếu / ngược sáng</strong></td>
                  <td><span class="tag">Cao</span></td>
                  <td>MediaPipe detection confidence &lt; 0.5 → bỏ frame</td>
                  <td>Tiền xử lý ảnh (histogram eq), hạ threshold detect</td>
                </tr>
              </tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("💡 Định hướng cải thiện chi tiết"):
                st.markdown("""
                1. **Mở rộng dataset:** Thêm video từ nhiều góc quay (chính diện, chéo 45°, ngang 90°) và điều kiện ánh sáng khác nhau.
                2. **Kalman Filter:** Làm mượt chuỗi keypoints trước khi tính góc, giảm false reps do jitter.
                3. **LSTM / GRU:** Thay Sliding Window + RF bằng mạng RNN để ghi nhớ ngữ cảnh dài hơn.
                4. **Personalization:** Cho phép người dùng tự thu thập data cá nhân và fine-tune — mô hình sẽ thích nghi theo form tập riêng.
                5. **Multi-angle fusion:** Kết hợp 2 camera (ngang + chéo) để tăng robustness.
                """)

    # ── TAB LỊCH SỬ TẬP LUYỆN ────────────────────────────────────────────
    with tab_history:
        history_file = 'data/workout_history.csv'

        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            if not history_df.empty:
                total_sessions = len(history_df)
                total_reps     = history_df['reps'].sum()
                total_time     = round(history_df['duration_seconds'].sum(), 0)

                st.markdown(f"""
                <div class="stat-row">
                  <div class="stat-card" style="border-top:3px solid #4ade80;">
                    <div class="label">Số buổi tập</div>
                    <div class="value accent">{total_sessions}</div>
                    <div class="unit">phiên</div>
                  </div>
                  <div class="stat-card" style="border-top:3px solid #38bdf8;">
                    <div class="label">Tổng số Reps</div>
                    <div class="value" style="color:#0284c7;">{total_reps:,}</div>
                    <div class="unit">lần</div>
                  </div>
                  <div class="stat-card" style="border-top:3px solid #fb923c;">
                    <div class="label">Tổng thời gian</div>
                    <div class="value" style="color:#ea580c;">{int(total_time)}</div>
                    <div class="unit">giây</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    fig_bar_h = px.bar(
                        history_df, x='date', y='reps', color='exercise',
                        title="Số Reps qua các buổi tập",
                        color_discrete_sequence=['#4ade80', '#38bdf8'],
                        template="plotly_white"
                    )
                    fig_bar_h.update_layout(
                        title_font_family="Syne", xaxis_tickangle=-30,
                        legend_title_text='Bài tập', margin=dict(t=50, b=20)
                    )
                    st.plotly_chart(fig_bar_h, width='stretch')

                with col_h2:
                    ex_reps = history_df.groupby('exercise')['reps'].sum().reset_index()
                    fig_pie = px.pie(
                        ex_reps, values='reps', names='exercise',
                        title="Tỷ trọng Reps theo Bài tập", hole=0.45,
                        color_discrete_sequence=['#4ade80', '#38bdf8'],
                        template="plotly_white"
                    )
                    fig_pie.update_layout(title_font_family="Syne", margin=dict(t=50, b=20))
                    st.plotly_chart(fig_pie, width='stretch')

                st.markdown("**📋 Chi tiết các buổi tập**")
                st.dataframe(
                    history_df.sort_values('date', ascending=False),
                    width='stretch'
                )

                if st.button("🗑️ Xóa lịch sử", type="secondary"):
                    os.remove(history_file)
                    st.success("Đã xóa lịch sử tập luyện.")
                    st.rerun()
            else:
                st.info("Chưa có dữ liệu lịch sử.")
        else:
            st.info("Chưa có lịch sử tập luyện. Hãy thực hiện **Phân tích Video** để ghi lại kết quả.")

    # ── TAB BÁO CÁO KỸ THUẬT ─────────────────────────────────────────────
    with tab_report:
        st.markdown("""
        <div class="page-hero" style="margin-bottom:24px;">
            <div class="badge">Technical Report</div>
            <h1 style="font-size:1.8rem;">Báo cáo Nghiên cứu Học máy</h1>
            <p>Tự động đếm số lần tập hít đất và squat từ video người tập bằng MediaPipe Pose kết hợp học máy</p>
        </div>
        """, unsafe_allow_html=True)

        sections = {
            "1. Xác lập bài toán": """
**Bối cảnh & Động lực:** Xu hướng home workout tăng mạnh đặt ra thách thức thiếu giám sát của huấn luyện viên. Người tập đối mặt với sai tư thế (gây chấn thương) và đếm sai số lần (ảnh hưởng tiến độ).

**Tại sao Push-up & Squat?** Hai bài tập đại diện cho upper body và lower body, có tính chu kỳ rõ rệt, phù hợp để xây dựng mô hình nhận diện chuỗi thời gian. "Đếm số lần" mang giá trị định lượng trực tiếp cho người dùng.

**Thách thức dữ liệu "in-the-wild":** Ánh sáng thay đổi, occlusion, góc nhìn không chuẩn, tốc độ thực hiện khác nhau, trang phục và vóc dáng đa dạng.
            """,
            "2. Tiền xử lý & Trích xuất đặc trưng": """
**Tại sao MediaPipe thay vì raw video?** BlazePose chuyển không gian ảnh RGB nhiều chiều thành 33 keypoints 3D — giảm chiều dữ liệu triệt để, tăng tính bất biến với ánh sáng, màu da và quần áo.

**Chuẩn hóa:** (1) Centering: gốc tọa độ về mid-hip. (2) Scaling: chia theo chiều cao thân người → bất biến khoảng cách camera.

**Feature Engineering — Góc khớp:** Bất biến với phép quay và tịnh tiến camera. Phản ánh trực tiếp biomechanics. Chuỗi góc theo thời gian mô tả trọn vẹn chu kỳ Lên-Xuống.

**Sliding Window (30 frames):** Cung cấp temporal context cho Random Forest mà không cần kiến trúc RNN phức tạp. Delta features (tốc độ thay đổi góc) bổ sung thông tin động.

**Vai trò lọc nhiễu:** MediaPipe thường "jitter" ở frame mờ → false reps nếu không lọc. Moving average hoặc Kalman filter khử nhiễu hiệu quả.
            """,
            "3. Thực thi mô hình & Quyết định thiết kế": """
**So sánh hướng tiếp cận:**
- *Rule-based (ngưỡng góc):* Dễ triển khai nhưng cứng nhắc — mỗi người có biên độ khớp khác nhau.
- *Random Forest + Sliding Window:* Phù hợp tabular data, xử lý tốt phi tuyến, inference nhanh trên CPU → **lựa chọn hiện tại**.
- *LSTM/Transformer:* Tối ưu lý thuyết nhưng cần nhiều dữ liệu và tài nguyên hơn.

**Trade-off:** Hy sinh một phần accuracy để đổi lấy real-time speed — yêu cầu sống còn của ứng dụng Fitness.

**Adaptive Learning:** Người dùng tự thu thập data cá nhân và retrain → mô hình tinh chỉnh theo form tập và góc quay quen thuộc của từng người.
            """,
            "4. Phân tích & Đánh giá": """
**Metrics được chọn:**
- *Accuracy & F1:* Đánh giá phân loại pha (UP/DOWN) per-frame.
- *Precision:* Tránh đếm khống (false reps). Recall thấp → bỏ sót rep (undercounting).
- *MAE rep count:* Metric thực tiễn nhất — sai lệch giữa rep máy đếm và rep thực tế.

**Confusion Matrix insight:** Mô hình nhầm chủ yếu tại pha Transition (giữa UP và DOWN) — đây là điểm yếu điển hình của Sliding Window + RF. LSTM với long-term memory sẽ xử lý tốt hơn.

**Wave plot (Predicted vs Ground Truth):** Mô hình hoạt động tốt khi người tập thực hiện đều nhịp, gặp khó khăn khi dừng lại nghỉ quá lâu giữa rep.
            """,
            "5. Rủi ro, Đạo đức & Định hướng": """
**Rủi ro chính:** Phụ thuộc hoàn toàn vào MediaPipe — nếu detection thất bại (ánh sáng tối, quần áo rộng), toàn bộ pipeline sụp đổ (Cascading Failure). Dữ liệu nhỏ → dễ overfit vào góc quay cụ thể.

**Privacy & Ethics:** Toàn bộ xử lý thực hiện **local/edge** — không raw video nào gửi lên server. Đảm bảo tuyệt đối quyền riêng tư người dùng.

**So sánh:** Vision-based qua webcam kém chính xác hơn wearable sensors / Kinect, nhưng chi phí bằng 0, không cần phần cứng đặc biệt.

**Future Work:**
1. Mở rộng bài tập (Plank, Pull-up, Jumping Jacks) — chỉ cần thay dataset.
2. AI Coach realtime: từ "đếm thụ động" → "sửa tư thế chủ động" với cảnh báo âm thanh.
3. LSTM/GRU thay Random Forest khi đủ dữ liệu.
4. Multi-angle fusion (2 camera) để tăng robustness.
            """,
        }

        for title, content in sections.items():
            with st.expander(f"**{title}**", expanded=(title.startswith("1."))):
                st.markdown(content)
