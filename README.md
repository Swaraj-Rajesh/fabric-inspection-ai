# 🧠 AI-Based Fabric Inspection System

This project implements a real-time fabric inspection system using AI, computer vision, and Raspberry Pi GPIO control. It detects major fabric defects, stops the motorized system, and activates a pump to mark the defective region.

## 🚀 Features
- Real-time fabric defect detection using OpenCV
- Motor control via Raspberry Pi GPIO
- TensorFlow-based AI inference
- PiCamera and USB camera support
- Visual alerts and motor stop with marking system

## 🧩 Folder Structure
```
fabric-inspection-ai/
├── models/           # Trained model (ai.h5)
├── src/              # Source code
├── images/           # (Optional) Sample images
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # Project description
```

## ▶️ How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place your `ai.h5` model in `models/`
4. Run: `python src/fabric_inspection.py`

## 🔧 Hardware Used
- Raspberry Pi
- PiCamera or USB webcam
- Stepper Motor + Driver
- Diaphragm Pump

## 📜 License
MIT License
