import os
import pandas as pd
import numpy as np

DATA_FILE = 'data/real_dataset.csv'

def save_video_data(angles_list, exercise_type, phase_label):
    """Save extracted angles from video to CSV."""
    os.makedirs('data', exist_ok=True)
    
    new_data = []
    for angles in angles_list:
        new_data.append({
            'angle_elbow': angles['angle_elbow'],
            'angle_hip': angles['angle_hip'],
            'angle_knee': angles['angle_knee'],
            'exercise_type': exercise_type,
            'phase_label': phase_label
        })
        
    df_new = pd.DataFrame(new_data)
    
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    df_combined['frame_id'] = range(len(df_combined))
    df_combined.to_csv(DATA_FILE, index=False)
    return len(df_new)

def load_real_data():
    """Load the real dataset."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

def normalize_angles(df):
    """Normalize angles to [0, 1] range."""
    df_norm = df.copy()
    angle_cols = ['angle_elbow', 'angle_hip', 'angle_knee']
    for col in angle_cols:
        df_norm[col] = df_norm[col] / 180.0
    return df_norm

def preprocess_sequence(angles_df, window_size=30):
    """Create sliding window features."""
    features = []
    labels = []
    
    angle_cols = ['angle_elbow', 'angle_hip', 'angle_knee']
    
    for i in range(len(angles_df) - window_size):
        window = angles_df.iloc[i:i+window_size]
        
        # Flatten window for traditional ML
        flat_features = window[angle_cols].values.flatten()
        
        # Add delta features (difference between first and last frame in window)
        delta = window[angle_cols].iloc[-1].values - window[angle_cols].iloc[0].values
        
        combined_features = np.concatenate([flat_features, delta])
        features.append(combined_features)
        
        # Label is the phase of the last frame in the window
        labels.append(window['phase_label'].iloc[-1])
        
    return np.array(features), np.array(labels)
