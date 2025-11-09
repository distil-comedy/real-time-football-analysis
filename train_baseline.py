# train_baseline.py
from ultralytics import YOLO
import os
import torch

def train_baseline():
    print("🚀 Starting baseline YOLOv8 training...")
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='E:/MyProjects/football_dataset/football.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        save=True,
        device=device,
        workers=4,
        lr0=0.01,  # learning rate
        weight_decay=0.0005,
        degrees=10,  # augmentation: rotation
        translate=0.1,  # augmentation: translation
        scale=0.5,  # augmentation: scale
        fliplr=0.5,  # augmentation: horizontal flip
    )
    
    print("✅ Training completed!")
    return results

if __name__ == "__main__":
    train_baseline()