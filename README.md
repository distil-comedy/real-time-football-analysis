<h1 align="center">⚽ Football Match Analysis System</h1>
<p align="center"> <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2VtbDZ0dzQ1Zzd4cnY3dDdnOXZ2azl3OGwwNjh4ZHJramlwOTNhZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/FoVzfcqCDSb7zCynOp/giphy.gif" width="150" alt="football"> </p> <h3 align="center">A Computer Vision System for Automated Football Match Analysis</h3>
<h2>📌 Overview</h2>

This project uses YOLOv8, DeepSORT, and a custom event logic pipeline to analyze football match footage and extract structured insights.
The system detects players, referees, goalposts, and the ball, tracks movement across frames, logs attributes, and visualizes analytics through an interactive Streamlit dashboard.

<h2>✨ Key Features</h2>

YOLOv8-Based Detection

DeepSORT Tracking with Persistent Player IDs

Ball Detection Logic

Event Recognition Pipeline

Attribute Extraction (actions, orientation, roles, speeds)

Dashboard for Match Analytics

Frame-Level Logging & Summary Reports

<h2>🛠️ Tech Stack</h2>
<p align="left"> <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="40" /> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" width="40" /> <img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/yolo-blue.png" width="40" /> <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="40" /> <img src="https://cvat.org/images/logo.svg" width="40" /> </p>


<h2>📸 Sample Output</h2>

Add snapshot images, detection screenshots, or dashboard pics here.

<h2>📊 Results</h2>

Total Frames Processed: 15,402

Best mAP50: 0.2292

Precision: 0.96

Integrated YOLO + DeepSORT for stable player tracking

Full pipeline generates detection logs, tracking logs, and a summary report

<h2>💡 Future Enhancements</h2>

Improve ball detection recall

Add action classification (passing, shooting, running)

Team color clustering

Heatmap and formation analysis

<h2>📜 License</h2>

This project is for academic and research purposes.
