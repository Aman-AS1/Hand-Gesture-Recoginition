# ✋ Hand Gesture Recognition Using MediaPipe (Python)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.1-orange.svg)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-Enabled-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-3.4.2%2B-blue.svg)

Real-time hand pose estimation using **MediaPipe Hands** and gesture recognition with a lightweight **MLP (TFLite)** model.

## 🌟 Features

- **Static hand-sign recognition** - Recognize hand poses and gestures
- **Dynamic finger-gesture recognition** - Track motion history for dynamic gestures
- **Custom data collection** - Easy logging system for training custom gestures
- **Lightweight & fast** - Optimized TFLite models for real-time performance
- **Retraining support** - Jupyter notebooks included for model customization

---

## ✅ Requirements

- Python 3.7+
- mediapipe 0.8.1
- OpenCV 3.4.2+
- TensorFlow 2.3.0+
- _(Optional)_ scikit-learn & matplotlib for confusion matrix

### Installation

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
hand-gesture-recognition/
├── app.py                                    # Main application
├── keypoint_classification.ipynb             # Train static gesture classifier
├── point_history_classification.ipynb        # Train dynamic gesture classifier
├── requirements.txt                          # Python dependencies
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint.csv                     # Training data for static gestures
│   │   ├── keypoint_classifier_label.csv    # Class labels
│   │   └── keypoint_classifier.tflite       # Trained model
│   └── point_history_classifier/
│       ├── point_history.csv                # Training data for dynamic gestures
│       ├── point_history_classifier_label.csv
│       └── point_history_classifier.tflite
└── utils/
    └── cvfpscalc.py                         # FPS calculation utility
```

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd hand-gesture-recognition
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   python app.py
   ```

4. **Show your hand to the camera** and see real-time gesture recognition!

---

## ⌨️ Controls

| Key   | Action                                        |
| ----- | --------------------------------------------- |
| `ESC` | Exit application                              |
| `n`   | Normal mode (default)                         |
| `k`   | Keypoint logging mode (static gestures)       |
| `h`   | Point history logging mode (dynamic gestures) |
| `0–9` | Assign class label when logging               |

---

## 🎓 Training Custom Gestures

### Static Hand Signs

1. **Collect training data**

   - Run `app.py`
   - Press `k` to enter keypoint logging mode
   - Perform your gesture and press `0–9` to assign a class
   - Data is saved to `model/keypoint_classifier/keypoint.csv`

2. **Train the model**

   - Open `keypoint_classification.ipynb` in Jupyter
   - Update `NUM_CLASSES` if you have more than default gestures
   - Run all cells to train and export the TFLite model
   - Update `model/keypoint_classifier/keypoint_classifier_label.csv` with your class names

3. **Default classes:**
   - `0` = Open hand
   - `1` = Close (fist)
   - `2` = Pointing

### Dynamic Finger Gestures

1. **Collect motion data**

   - Run `app.py`
   - Press `h` to enter history logging mode
   - Perform your dynamic gesture and press `0–9`
   - Data is saved to `model/point_history_classifier/point_history.csv`

2. **Train the model**

   - Open `point_history_classification.ipynb`
   - Update `NUM_CLASSES` if needed
   - Run all cells to train and export
   - Update the label CSV file

3. **Default classes:**
   - `0` = Stationary
   - `1` = Clockwise rotation
   - `2` = Counter-clockwise rotation
   - `4` = Moving

---

## 🧠 How It Works

1. **MediaPipe** detects 21 hand landmarks in real-time
2. Landmarks are converted to **normalized relative coordinates**
3. Features are **flattened** and fed to an MLP classifier
4. **Static gestures** use instantaneous hand pose
5. **Dynamic gestures** use fingertip trajectory over time (motion history)

### Architecture

```
Camera Input → MediaPipe Hands → 21 Landmarks → Normalization → MLP Classifier → Gesture Label
```

---

## 📊 Model Details

- **Framework:** TensorFlow Lite
- **Architecture:** Multi-layer Perceptron (MLP)
- **Input:** Normalized hand landmark coordinates
- **Output:** Gesture class probability distribution
- **Optimization:** Quantized for edge deployment

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🪪 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for hand tracking
- [TensorFlow Lite](https://www.tensorflow.org/lite) for efficient model deployment
- OpenCV community for computer vision tools

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ using MediaPipe and TensorFlow**
