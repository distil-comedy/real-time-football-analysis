# analyze_dataset.py
import os
import yaml
from collections import Counter

def analyze_annotations():
    dataset_path = "E:/MyProjects/football_dataset"
    
    # Load class names from YAML
    yaml_path = os.path.join(dataset_path, "football.yaml")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    
    # Analyze training labels
    labels_path = os.path.join(dataset_path, "labels/train")
    class_counts = Counter()
    total_objects = 0
    
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        total_objects += 1
    
    print("📊 Dataset Analysis Results:")
    print(f"Total objects in training set: {total_objects}")
    print("\nClass distribution:")
    for class_id, count in class_counts.most_common():
        class_name = class_names.get(class_id, f"Unknown_{class_id}")
        percentage = (count / total_objects) * 100
        print(f"  {class_name} (ID {class_id}): {count} objects ({percentage:.1f}%)")
    
    return class_counts

if __name__ == "__main__":
    analyze_annotations()