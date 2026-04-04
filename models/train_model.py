import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import joblib
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import load_real_data, normalize_angles, preprocess_sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def augment_data(df, noise_level=5.0):
    """Add random noise to angles for data augmentation."""
    augmented = df.copy()
    for col in ['angle_elbow', 'angle_shoulder', 'angle_hip', 'angle_knee']:
        if col in augmented.columns:
            noise = np.random.normal(0, noise_level, len(augmented))
            augmented[col] = augmented[col] + noise
            augmented[col] = augmented[col].clip(0, 180)
    return augmented

def train_models(n_estimators=100, max_depth=None, use_augmentation=False):
    """Train Random Forest model on real video data."""
    df = load_real_data()
    if df is None or len(df) < 100:
        return False, "Không đủ dữ liệu để huấn luyện. Vui lòng thu thập thêm video (ít nhất 100 frames)."
        
    if use_augmentation:
        df_aug = augment_data(df)
        df = pd.concat([df, df_aug], ignore_index=True)
        
    df_norm = normalize_angles(df)
    window_size = 30
    
    if len(df_norm) <= window_size:
        return False, f"Dữ liệu quá ít. Cần nhiều hơn {window_size} frames."
        
    X, y = preprocess_sequence(df_norm, window_size=window_size)
    
    if len(X) < 10:
        return False, "Không đủ chuỗi dữ liệu (sequences) sau khi tiền xử lý."
        
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    if len(np.unique(y_encoded)) < 2:
        return False, "Cần ít nhất 2 nhãn khác nhau (VD: UP và DOWN) để huấn luyện."
        
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    rf_preds = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    joblib.dump(rf_model, 'models/exercise_model.pkl')
    
    metrics = {
        'accuracy': float(rf_acc),
        'f1_score': float(f1_score(y_test, rf_preds, average='weighted')),
        'precision': float(precision_score(y_test, rf_preds, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, rf_preds).tolist(),
        'classes': le.classes_.tolist()
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return True, f"Huấn luyện thành công! Độ chính xác trên tập test: {rf_acc*100:.2f}%"

if __name__ == "__main__":
    success, msg = train_models()
    print(msg)
