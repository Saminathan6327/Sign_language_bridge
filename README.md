# Language Bridge

Real-time sign language interpreter vision pipeline and interactive accessibility kiosk built with Python, OpenCV, MediaPipe Tasks Vision, and JavaScript.

## Overview

Language Bridge is an accessibility-focused web application that detects hand gestures through a webcam and converts them into readable text and spoken responses. It supports real-time dual-hand tracking, gesture classification, interactive chat interaction, text-to-speech output, and custom machine learning model training.

## Features

- **Dual-Hand Landmark Tracking (42 Keypoints)**: Tracks both hands simultaneously (21 3D points per hand = 42 points total) with distinct visual color feedback.
- **Real-Time Gesture Calculation Engine**: High-speed geometric & kinematic recognizer that calculates both **two-hand gestures** and **single-hand gestures**.
- **Interactive Accessibility Chatbox**: Real-time sign language conversation feed featuring:
  - Live staging buffer with hold-to-send progress indicator.
  - User Signer message bubbles with gesture emoji, confidence score, and hand count badge.
  - Automated AI Counter Assistant responses providing contextual replies.
  - Text-to-Speech (TTS) readout powered by the Web Speech API.
  - Manual text input, message editing, and one-click chat transcript export.
  - Quick-access gesture palette chips.
- **126-Coordinate Feature Payload**: Flattens `(x, y, z)` coordinates for up to 42 points into 126 floating-point values ready for machine learning inference and dataset recording.
- **Hybrid Inference & Telemetry**: Seamless bridge to Express (`POST /predict`) and Python `predict.py` with zero-latency client-side calculation fallback.
- **DirectShow Camera Acceleration**: Low-latency video capture (`CAP_DSHOW` on Windows) for high FPS (>45 FPS on CPU).

---

## Demo

![Language Bridge Demo](assets/demo.gif)

> *Place your preview GIF or screenshot at `assets/demo.gif` to display a preview here.*

---

## Requirements

- **Python**: 3.9 or newer
- **Node.js**: 18 or newer
- **Webcam**: Built-in or external USB camera
- **Operating System**: Windows, macOS, or Linux
- **Browser**: Modern web browser (Chrome, Edge, Firefox, Safari)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vairaprakash-06/Language_bridge.git
   cd Language_bridge
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

---

## Quick Start

### Terminal 1: Start the backend server
```bash
npm run server
```
The Express API server starts on `http://localhost:3000`.

### Terminal 2: Start the frontend interface
```bash
npm run dev
```
The Vite development server starts on `http://localhost:5173`.

### Access the Web App
Open `http://localhost:5173` in your browser and allow camera permissions when prompted.

---

## Supported Gestures

Language Bridge recognizes a wide variety of two-hand and single-hand gestures:

### Dual-Hand Gestures (42 Keypoints)

| Gesture | Label | Meaning / Context |
|---|---|---|
| **Heart** | `HEART` | Love, appreciation, or affection |
| **Stop / Cross** | `STOP` | Stop, wait, or crossed wrists |
| **Thank You / Namaste** | `THANK_YOU` | Greeting, gratitude, or respect |
| **Friend** | `FRIEND` | Interlinked index fingers |
| **We** | `WE` | Both index fingers pointing together |
| **Help** | `HELP` | Fist resting on flat open palm |
| **More** | `MORE` | Fingertips pinched together |
| **Together** | `TOGETHER` | Fists placed side-by-side |
| **Done / Double Thumbs Up** | `DONE` | Complete, approval, or finished |
| **Food** | `FOOD` | Open palms held side-by-side facing upward |
| **Please** | `PLEASE` | Open cupped palms held together |
| **Welcome** | `WELCOME` | Both open palms spread wide |

### Single-Hand Gestures (21 Keypoints)

| Gesture | Label | Meaning / Context |
|---|---|---|
| **I Love You** | `LOVE` | ASL I Love You sign (Thumb + Index + Pinky) |
| **OK / Good** | `GOOD` | Approval (Thumb and index circle) |
| **Water** | `WATER` | Water request (W handshape) |
| **Peace** | `PEACE` | Peace sign / V sign |
| **Yes** | `YES` | Thumbs up |
| **No** | `NO` | Thumbs down |
| **Hello** | `HELLO` | Open palm wave / greeting |
| **Bad** | `BAD` | Palm turned downward |
| **I / Me** | `I` | Pointing index finger |
| **Like** | `LIKE` | Liking something |
| **Sorry** | `SORRY` | Closed fist over chest |

---

## Training a Custom Model

You can collect landmark samples and train a custom SVM classifier:

1. **Collect gesture samples:**
   ```bash
   python collect_data.py --max-hands 2
   ```
   - Press keys `A`–`Z` or `0`–`9` to record landmark data for specific gesture labels into `landmarks.csv`.
   - Press `SPACEBAR` to record for the active label.
   - Press `q` or `ESC` to save progress and exit.

2. **Train the classifier:**
   ```bash
   python train.py --data landmarks.csv --output model.pkl
   ```

3. **Start the prediction service:**
   ```bash
   npm run server
   ```

> **Note:** `landmarks.csv` and `model.pkl` are generated files. Place `model.pkl` in the project root directory (`./model.pkl`), where `predict.py` and `server.js` will automatically load it.

---

## Configuration

The Express backend server can be configured using environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Port for the backend API server |
| `PYTHON_BIN` | `python` | Path or binary name for the Python executable |

---

## API Reference

### `POST /predict`

Sends landmark coordinates to the backend for gesture prediction.

#### Request Body
Accepts a JSON payload containing 63 coordinates (21 keypoints / 1 hand) or 126 coordinates (42 keypoints / 2 hands):

```json
{
  "landmarks": [0.12, 0.24, 0.31, 0.05, 0.18, 0.22]
}
```

#### Response Body
```json
{
  "prediction": "HELLO",
  "confidence": 0.94,
  "handsDetected": 1,
  "points": 21
}
```

### `POST /api/smooth-sentence`

Refines raw transcribed sign language words into natural conversational English or Tamil.

#### Request Body
```json
{
  "sentence": "HELP PLEASE WATER"
}
```

#### Response Body
```json
{
  "original": "HELP PLEASE WATER",
  "smoothedSentence": "Could you please help me? I need some water.",
  "model": "sign-bridge-llm-smoother-v1",
  "timestamp": "2026-10-05T22:08:28.000Z"
}
```

---

## Python Vision Pipeline (`vision.py`)

Run the standalone real-time vision pipeline with dual-hand tracking:

```bash
python vision.py --max-hands 2
```

### CLI Arguments
| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--width` | `640` | Video feed width |
| `--height` | `480` | Video feed height |
| `--max-hands` | `2` | Maximum hands to detect simultaneously (1 or 2) |

Press `q` or `ESC` in the video window to quit.

---

## Project Structure

```text
Language_bridge/
├── server.js          # Express API server (port 3000, CORS, /predict endpoint)
├── predict.py         # Lightweight Python inference worker (SVM model loader + 42-pt heuristic)
├── index.html         # Web kiosk interface (HTML5 video, canvas overlay, Chatbox)
├── main.js            # Frontend logic (MediaPipe Tasks Vision CDN, 42-pt tracking, Chatbox)
├── style.css          # Kiosk design system (Public service counter aesthetic)
├── package.json       # Project scripts and dependencies (Vite, Express, CORS)
├── collect_data.py    # Dataset collection tool (logs 42-pt / 21-pt landmarks to CSV)
├── train.py           # SVM classifier training script (scikit-learn, supports 63 & 126 feats)
├── vision.py          # Python vision pipeline & dual-hand landmark extractor
├── requirements.txt   # Python dependencies
├── landmarks.csv      # Generated dataset (coordinates + label)
├── model.pkl          # Trained SVM model bundle (model + LabelEncoder)
└── README.md          # Project documentation
```

> **Note:** `landmarks.csv` and `model.pkl` are generated files and may not be present in a fresh clone until data collection and model training steps are executed.

---

## Troubleshooting

### Camera is not detected
- Check that your camera is connected and not being used by another application.
- Allow camera permissions in the browser when prompted.
- Try changing the camera index:
  ```bash
  python vision.py --camera 1
  ```

### Backend connection fails
- Make sure the Express server is running in a separate terminal:
  ```bash
  npm run server
  ```
- Verify that port `3000` is open and available.

### Predictions are inaccurate
- Collect more samples using `collect_data.py`.
- Use consistent, adequate lighting.
- Keep your hands inside the camera frame.
- Retrain the model using a balanced dataset via `python train.py`.

---

## Limitations

- Recognition accuracy depends on lighting, background contrast, and camera resolution.
- The model currently supports the predefined single-hand and dual-hand gestures in the dataset/heuristic engine.
- Fast hand movements or partial hand occlusion may reduce real-time tracking precision.
- A physical webcam is required for real-time video interpretation.

---

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/improved-gesture-detection
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/improved-gesture-detection
   ```
5. Open a Pull Request.

---

## Contributors

Thanks to the following contributors for building and maintaining Language Bridge:

| Avatar | Contributor | GitHub Profile | Role |
|:---:|---|---|---|
| <img src="https://github.com/vairaprakash-06.png?size=60" width="50px;" style="border-radius:50%;" alt="Saminathan Muruganantham" /> | **Saminathan Muruganantham** | [@vairaprakash-06](https://github.com/vairaprakash-06) / [@Saminathan6327](https://github.com/Saminathan6327) | Project Author & Lead Maintainer |

---

## License

This project is licensed under the **MIT License**.

---

## Author

Created with ❤️ by [Saminathan Muruganantham (vairaprakash-06)](https://github.com/vairaprakash-06).

