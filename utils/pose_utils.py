import cv2
import numpy as np
import mediapipe as mp

class PoseProcessor:
    """Class to process video frames and extract pose landmarks using MediaPipe."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_keypoints(self, frame):
        """Extract 33 keypoints from a frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results.pose_landmarks
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between 3 points (in degrees)."""
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
        
    def get_exercise_angles(self, landmarks):
        """Return a dictionary of relevant angles for exercises."""
        if not landmarks:
            return None
            
        # Extract coordinates
        landmarks_list = landmarks.landmark
        
        # Get coordinates for left side (assuming side view)
        shoulder = [landmarks_list[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks_list[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks_list[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks_list[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks_list[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks_list[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks_list[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks_list[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks_list[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks_list[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks_list[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks_list[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                 
        # Calculate angles
        angle_elbow = self.calculate_angle(shoulder, elbow, wrist)
        angle_shoulder = self.calculate_angle(hip, shoulder, elbow)
        angle_hip = self.calculate_angle(shoulder, hip, knee)
        angle_knee = self.calculate_angle(hip, knee, ankle)
        
        return {
            'angle_elbow': angle_elbow,
            'angle_shoulder': angle_shoulder,
            'angle_hip': angle_hip,
            'angle_knee': angle_knee
        }
        
    def classify_state(self, angles, exercise_type):
        """Simple heuristic classification (fallback if ML model not used)."""
        if not angles:
            return "UNKNOWN"
            
        if exercise_type == "Push-up":
            if angles['angle_elbow'] < 90:
                return "DOWN"
            elif angles['angle_elbow'] > 160:
                return "UP"
            else:
                return "TRANSITION"
                
        elif exercise_type == "Squat":
            if angles['angle_knee'] < 90 and angles['angle_hip'] < 100:
                return "DOWN"
            elif angles['angle_knee'] > 160 and angles['angle_hip'] > 160:
                return "UP"
            else:
                return "TRANSITION"
                
        return "UNKNOWN"
        
    def get_feedback(self, angles, exercise_type, current_state):
        """Provide real-time feedback based on angles and current state."""
        if not angles:
            return ""
            
        feedback = "Tốt"
        if exercise_type == "Push-up":
            if current_state == "DOWN" and angles['angle_elbow'] > 100:
                feedback = "Xuống sâu hơn"
            elif current_state == "UP" and angles['angle_elbow'] < 150:
                feedback = "Lên thẳng tay"
            elif angles['angle_hip'] < 150:
                feedback = "Giữ thẳng lưng"
                
        elif exercise_type == "Squat":
            if current_state == "DOWN" and angles['angle_knee'] > 100:
                feedback = "Ngồi thấp xuống"
            elif current_state == "UP" and angles['angle_knee'] < 150:
                feedback = "Đứng thẳng lên"
            elif angles['angle_hip'] < 80:
                feedback = "Đừng gập người quá nhiều"
                
        return feedback
        
    def draw_overlay(self, frame, landmarks, state, count, exercise_type="Exercise", feedback=""):
        """Draw skeleton and text information on the frame."""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
        # Draw status box
        cv2.rectangle(frame, (0,0), (350, 120), (245, 117, 16), -1)
        
        # Display Exercise
        cv2.putText(frame, 'EXERCISE', (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, exercise_type, (15, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Display Reps
        cv2.putText(frame, 'REPS', (15, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(count), (15, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Display State
        cv2.putText(frame, 'STATE', (120, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, state, (120, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Display Feedback
        if feedback:
            color = (0, 255, 0) if feedback == "Tốt" else (0, 0, 255)
            cv2.putText(frame, 'FEEDBACK', (220, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, feedback, (220, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    
        return frame
