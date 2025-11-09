# ball_detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class BallDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def color_based_detection(self, frame):
        """Complementary color-based ball detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color ranges for football (adjust based on your ball)
        # White range
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        
        # Yellow range  
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Create masks
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Ball size range
                x, y, w, h = cv2.boundingRect(contour)
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                
                # Check if contour is circular
                if circularity > 0.6:
                    ball_candidates.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'confidence': min(circularity, 1.0)
                    })
        
        return ball_candidates
    
    def detect_ball_in_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # YOLO detection
            yolo_results = self.model(frame, conf=0.3, classes=[3])  # Class 3 is ball
            
            # Color-based detection
            ball_candidates = self.color_based_detection(frame)
            
            # Combine results
            annotated_frame = yolo_results[0].plot()
            
            # Draw color-based detections
            for candidate in ball_candidates:
                x, y, w, h = candidate['bbox']
                confidence = candidate['confidence']
                
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Ball {confidence:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow("Ball Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = BallDetector("runs/detect/train/weights/best.pt")
    detector.detect_ball_in_video("videos/match_video.mp4")