⚽ Football Match Analysis System
A compact computer-vision pipeline for detecting players, referees and the ball in football footage, tracking their movement, recognizing events and visualizing analytics through an interactive dashboard.

✨ Features

YOLOv8-based detection

DeepSORT multi-object tracking

Ball tracking heuristics

Event recognition (passes, possession, shots, ball-out)

Streamlit dashboard for analytics

Modular preprocessing, detection and tracking pipeline

Exportable processed video and logs

📦 Tech Stack

Python

YOLOv8

DeepSORT

OpenCV

Streamlit

CVAT

📁 Project Structure
root/
│── src/
│   ├── detection/
│   ├── tracking/
│   ├── events/
│   ├── preprocessing/
│   ├── utils/
│── dashboard/
│── models/
│── outputs/
│── requirements.txt
│── README.md

▶️ Running the System
1. Install Requirements
pip install -r requirements.txt

2. Run Detection & Tracking
python src/main.py --video input.mp4

3. Launch Dashboard
streamlit run dashboard/app.py

📜 License

Academic & research use.
