"""
vision.py - Real-Time Sign Language Interpreter Vision Pipeline

Features:
- Captures live webcam feed using OpenCV.
- Initializes MediaPipe Hands (supports modern Tasks API and legacy Solutions API).
- Draws 21 hand landmarks on the video frame in real-time.
- Provides `extract_landmarks()` helper function to extract 63 flattened (x, y, z) features.
- Highly optimized for frame rate (minimal latency, low-overhead inference).
- Clean exit and resource management tailored for VS Code.
"""

import argparse
import os
import sys
import time
import urllib.request

# Suppress low-level MediaPipe / TensorFlow C++ logging in VS Code terminal
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import mediapipe as mp

# Default model URL and file for MediaPipe Tasks API
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model_downloaded(model_path: str = MODEL_PATH) -> str:
    """
    Ensures that the MediaPipe Hand Landmarker model file exists.
    Downloads it automatically from Google's repository if missing.
    """
    if not os.path.exists(model_path):
        print(f"[*] Downloading MediaPipe hand landmark model to '{model_path}'...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[*] Model download complete.")
    return model_path


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Extracts the (x, y, z) coordinates of the 21 MediaPipe hand landmarks 
    and flattens them into a single 1D NumPy array (63 features total).

    Args:
        hand_landmarks: 
            Either a MediaPipe NormalizedLandmarkList (legacy solutions API)
            or a list of NormalizedLandmark objects (modern Tasks API).
            If None, returns an array of 63 zeros.

    Returns:
        np.ndarray: 1D array of shape (63,) with dtype float32:
                    [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
    """
    if hand_landmarks is None:
        return np.zeros(21 * 3, dtype=np.float32)

    # Support both legacy solutions (object with .landmark attribute) and Tasks API (list)
    landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)

    # Fast flattening using list comprehension
    coords = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]

    # Guarantee exactly 63 elements
    if len(coords) != 63:
        features = np.zeros(63, dtype=np.float32)
        features[:min(len(coords), 63)] = coords[:63]
        return features

    return np.array(coords, dtype=np.float32)


def extract_dual_landmarks(hand_landmarks_list, max_hands: int = 2) -> np.ndarray:
    """
    Extracts (x, y, z) coordinates for up to 2 hands (42 points total = 126 floats).
    - If 2 hands are detected: 21 points for Hand 1 + 21 points for Hand 2 = 42 points.
    - If 1 hand is detected: 21 points for Hand 1 + 21 zeroed points for Hand 2 = 42 points.
    - If 0 hands: 42 zeroed points (126 floats).

    Returns:
        np.ndarray: 1D array of shape (126,) with dtype float32.
    """
    total_features = max_hands * 21 * 3  # 126 floats
    features = np.zeros(total_features, dtype=np.float32)

    if not hand_landmarks_list:
        return features

    for h_idx in range(min(len(hand_landmarks_list), max_hands)):
        hand = hand_landmarks_list[h_idx]
        single_hand_feats = extract_landmarks(hand)
        start = h_idx * 63
        features[start : start + 63] = single_hand_feats

    return features


class SignLanguageHandDetector:
    """
    High-performance Hand Detector with dual-engine support:
    - MediaPipe Tasks API (modern 0.10.14+ / 1.0+)
    - MediaPipe Solutions API (legacy <0.10.14)
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_tasks_api = not (hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"))
        self._timestamp_ms = 0

        if self.use_tasks_api:
            # Modern MediaPipe Tasks API
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections

            model_file = ensure_model_downloaded(MODEL_PATH)
            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.drawing_utils = drawing_utils
            self.connections = HandLandmarksConnections.HAND_CONNECTIONS
        else:
            # Legacy MediaPipe Solutions API
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = getattr(mp.solutions, "drawing_styles", None)
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Processes a BGR image frame, detects hands, and returns detected hands list.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            # MediaPipe Tasks API expects mp.Image and monotonic timestamp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self._timestamp_ms += 33  # Approx increment per frame
            result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
            return result.hand_landmarks  # List of hands, each is list of NormalizedLandmark
        else:
            # MediaPipe Solutions API
            # Optimization: Mark image as not writeable to pass by reference
            frame_rgb.flags.writeable = False
            results = self.detector.process(frame_rgb)
            frame_rgb.flags.writeable = True
            return results.multi_hand_landmarks if results.multi_hand_landmarks else []

    def draw_landmarks_on_frame(self, frame_bgr: np.ndarray, hand_landmarks_list) -> None:
        """
        Draws the 21 landmarks per hand with distinct styling for Hand 1 vs Hand 2.
        """
        if not hand_landmarks_list:
            return

        h, w, _ = frame_bgr.shape
        colors = [
            ((255, 200, 0), (0, 255, 128)),   # Hand 1: Cyan connections, Emerald joints
            ((0, 165, 255), (0, 255, 255)),   # Hand 2: Sunset Amber connections, Yellow joints
        ]

        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            conn_color, pt_color = colors[idx % len(colors)]

            if self.use_tasks_api:
                self.drawing_utils.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.connections
                )
            else:
                landmark_style = self.mp_drawing.DrawingSpec(color=pt_color, thickness=2, circle_radius=3)
                connection_style = self.mp_drawing.DrawingSpec(color=conn_color, thickness=2, circle_radius=2)

                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style
                )

            # Draw Hand Label badge near wrist landmark (point 0)
            landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
            if landmarks and len(landmarks) > 0:
                wrist = landmarks[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                label_text = f"Hand {idx + 1} (21 pts)"
                cv2.rectangle(frame_bgr, (wx - 4, wy - 22), (wx + 110, wy + 2), (20, 20, 20), -1)
                cv2.putText(
                    frame_bgr,
                    label_text,
                    (wx, wy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    pt_color,
                    1,
                    cv2.LINE_AA,
                )

    def close(self):
        """Releases internal resources cleanly."""
        if hasattr(self.detector, "close"):
            self.detector.close()


def run_pipeline(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    max_hands: int = 2
):
    """
    Main webcam capture and landmark processing loop.
    """
    window_name = "Real-Time Sign Language Interpreter - Vision Pipeline"

    # Optimize camera backend for Windows (CAP_DSHOW provides instant startup and low latency)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera with index {camera_index}.")
        print("[!] Please verify your webcam is connected or try a different index (e.g., --camera 1).")
        return

    # Optimize capture settings for frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency / buffering delay

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("Sign Language Vision Pipeline Initialized")
    print(f"- Resolution: {actual_w}x{actual_h}")
    print(f"- Camera Index: {camera_index}")
    print(f"- Max Hands: {max_hands} (42 Points Tracking)")
    print("- Press 'q' or 'ESC' in the video window to quit.")
    print("=" * 60)

    # Initialize hand detector
    detector = SignLanguageHandDetector(max_num_hands=max_hands)

    # FPS tracking variables
    prev_time = time.perf_counter()
    fps = 0.0
    alpha = 0.9  # Exponential smoothing factor for FPS display

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Warning: Empty frame received from webcam. Retrying...")
                time.sleep(0.01)
                continue

            # Mirror view for natural interaction (selfie-view)
            frame = cv2.flip(frame, 1)

            # Process frame through MediaPipe
            hands_landmarks = detector.process_frame(frame)

            # Draw the landmarks on the frame
            detector.draw_landmarks_on_frame(frame, hands_landmarks)

            # Extract dual-hand features (42 points = 126 coordinates)
            dual_features = extract_dual_landmarks(hands_landmarks, max_hands=max_hands)
            hands_count = len(hands_landmarks) if hands_landmarks else 0
            points_tracked = min(hands_count, 2) * 21

            # Calculate smoothed FPS
            curr_time = time.perf_counter()
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0:
                current_fps = 1.0 / delta
                fps = alpha * fps + (1.0 - alpha) * current_fps if fps > 0 else current_fps

            # UI HUD Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (360, 95), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Text information
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 128),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Hands: {hands_count}/2 (Points: {points_tracked}/42 | Feats: 126)",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'q' or 'ESC' to exit",
                (20, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Display frame
            cv2.imshow(window_name, frame)

            # Clean exit handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break

            # Handle window close button ('X') cleanly in VS Code
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        # Guarantee resources are released cleanly in VS Code / Windows
        print("[*] Releasing camera and closing windows...")
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Vision pipeline terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Interpreter - Vision Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--max-hands", type=int, default=2, help="Max hands to detect (default: 2)")
    args = parser.parse_args()

    run_pipeline(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        max_hands=args.max_hands,
    )
