# player_tracking.py
from ultralytics import YOLO
import cv2
import numpy as np

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.track_history = {}
        
    def track_video(self, video_path, output_path="tracking_output.mp4"):
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print("🎬 Starting player tracking...")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Track objects with YOLO
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml",  # Use ByteTrack for better performance
                conf=0.5,
                classes=[0, 1, 2]  # Track players, goalkeepers, referees
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Get tracking data
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                # Visualize tracking
                annotated_frame = results[0].plot()
                
                # Draw tracking trails
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    center = (int(x), int(y))
                    
                    # Initialize track history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # Store center points
                    self.track_history[track_id].append(center)
                    
                    # Keep only last 30 points
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)
                    
                    # Draw trail
                    if len(self.track_history[track_id]) > 1:
                        points = np.array(self.track_history[track_id], dtype=np.int32)
                        cv2.polylines(annotated_frame, [points], isClosed=False, 
                                    color=(0, 255, 0), thickness=2)
                    
                    # Add ID label
                    cv2.putText(annotated_frame, f"ID: {track_id}", 
                               (int(x - w/2), int(y - h/2 - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                out.write(annotated_frame)
                
                if frame_count % 30 == 0:
                    print(f"📹 Processed frame {frame_count}, tracking {len(track_ids)} objects")
            
            frame_count += 1
            
            # Display progress
            if frame_count % 100 == 0:
                cv2.imshow("Player Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Tracking completed! Output saved as {output_path}")

if __name__ == "__main__":
    tracker = PlayerTracker("runs/detect/train/weights/best.pt")
    tracker.track_video("videos/match_video.mp4")