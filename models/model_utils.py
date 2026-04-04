import os
import json
import joblib
import numpy as np

def load_model_cached():
    """Load the trained model and scaler with error handling."""
    try:
        model_path = 'models/exercise_model.pkl'
        scaler_path = 'models/scaler.pkl'
        le_path = 'models/label_encoder.pkl'
        
        if not os.path.exists(model_path):
            return None, None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        
        return model, scaler, le
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def load_metrics():
    """Load training metrics from JSON."""
    try:
        metrics_path = 'models/metrics.json'
        if not os.path.exists(metrics_path):
            return None
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

def predict_phase(model, scaler, le, angles_window):
    """Predict exercise phase from a window of angles."""
    if model is None or scaler is None or le is None:
        return "UNKNOWN", 0.0
        
    try:
        # Flatten window
        flat_features = angles_window.flatten()
        
        # Add delta
        delta = angles_window[-1] - angles_window[0]
        
        combined_features = np.concatenate([flat_features, delta])
        
        # Scale
        scaled_features = scaler.transform([combined_features])
        
        # Predict
        pred_idx = model.predict(scaled_features)[0]
        probs = model.predict_proba(scaled_features)[0]
        
        confidence = probs[pred_idx]
        phase = le.inverse_transform([pred_idx])[0]
        
        return phase, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "UNKNOWN", 0.0

def count_reps_from_predictions(predictions, exercise_type):
    """Count repetitions from a sequence of phase predictions."""
    reps = 0
    current_state = "UP"
    
    down_label = f"{exercise_type.lower()}_down"
    up_label = f"{exercise_type.lower()}_up"
    
    for pred in predictions:
        if pred == down_label and current_state == "UP":
            current_state = "DOWN"
        elif pred == up_label and current_state == "DOWN":
            current_state = "UP"
            reps += 1
            
    return reps

def get_model_info():
    """Return dictionary with model information."""
    metrics = load_metrics()
    if metrics:
        return {
            "status": "Trained",
            "accuracy": metrics.get("accuracy", 0),
            "classes": metrics.get("classes", [])
        }
    return {"status": "Not Trained"}
