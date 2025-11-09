# verify_dataset.py
import os
import yaml
from pathlib import Path

def verify_dataset():
    dataset_path = "E:/MyProjects/football_dataset"
    
    print("🔍 Verifying dataset structure...")
    
    # Check required directories
    required_dirs = [
        "images/train",
        "images/val", 
        "labels/train",
        "labels/val"
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing: {full_path}")
            all_dirs_exist = False
        else:
            file_count = len(os.listdir(full_path))
            print(f"✅ {dir_path}: {file_count} files")
    
    # Verify YAML configuration
    yaml_path = os.path.join(dataset_path, "football.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ YAML config loaded: {config['nc']} classes")
    else:
        print("❌ football.yaml not found")
        all_dirs_exist = False
    
    return all_dirs_exist

if __name__ == "__main__":
    if verify_dataset():
        print("🎉 Dataset verification completed successfully!")
    else:
        print("❌ Dataset has issues that need to be fixed.")