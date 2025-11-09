# test_baseline.py
from ultralytics import YOLO
import cv2
import os

def test_model():
    # Load the best trained model
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print("❌ Trained model not found. Run training first.")
        return
    
    model = YOLO(model_path)
    
    # Test on validation set
    results = model.val()
    print("📊 Validation results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    # Test on sample images
    test_images = "E:/MyProjects/football_dataset/images/val"
    sample_results = model.predict(
        source=test_images, 
        save=True, 
        conf=0.5,
        imgsz=640
    )
    
    print(f"✅ Predictions saved in runs/detect/predict/")

if __name__ == "__main__":
    test_model()