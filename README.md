# ✋ Hand Gesture Recognition Using MediaPipe (Python)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.1-orange.svg)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-Enabled-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-3.4.2%2B-blue.svg)

Real-time hand pose estimation using **MediaPipe Hands** and gesture recognition
with a lightweight **MLP (TFLite)** model.

Supports:

- Static hand-sign recognition
- Dynamic finger-gesture recognition (motion history)
- Custom data collection and retraining

---

## ✅ Requirements

- mediapipe 0.8.1
- OpenCV 3.4.2+
- TensorFlow 2.3.0+
- _(Optional)_ scikit-learn & matplotlib for confusion matrix

Install:

```bash
pip install -r requirements.txt
```

📂 Project Structure
app.py
keypoint_classification.ipynb
point_history_classification.ipynb

model/
├─ keypoint_classifier/
└─ point_history_classifier/

utils/
└─ cvfpscalc.py

Controls
Key Action
ESC Exit
n Normal mode
k Log keypoints
h Log point history
0–9 Assign class label

🎓 Training (Quick)
Static hand signs

Press k → logging mode

Press 0–9 → saves to model/keypoint_classifier/keypoint.csv

Open keypoint_classification.ipynb and run

Update NUM_CLASSES + label CSV if adding gestures

Default classes: 0=open, 1=close, 2=pointing
Dynamic finger gestures

Press h → history logging

Press 0–9 → saves to model/point_history_classifier/point_history.csv

Run point_history_classification.ipynb

Update NUM_CLASSES + label CSV if needed

Default classes: 0=stationary, 1=clockwise, 2=counter-clockwise, 4=moving

🧠 How It Works

MediaPipe detects 21 hand landmarks

Landmarks → normalized relative coordinates

Flattened features → fed to MLP classifier

Dynamic gestures use fingertip trajectory over time

🪪 License

MIT License
