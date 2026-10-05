"""
collect_data.py - Sign Language Dataset Collection Tool

Features:
- Captures live webcam feed using OpenCV.
- Detects 21 3D hand landmarks in real-time using MediaPipe Hands.
- Flattens the 21 3D landmarks into 63 floating-point numbers (x, y, z per landmark).
- Listens for keyboard inputs to record gestures (e.g., press 'A'-'Z' or '0'-'9' to record that label,
  or press SPACE to record for the current active label).
- Appends recorded samples to `landmarks.csv` with format: [label, x0, y0, z0, ..., x20, y20, z20].
- Renders an interactive real-time visual HUD showing:
  * Current session sample count (prominently displayed)
  * Breakdown of samples collected per label in this session
  * Total samples in the CSV file
  * Real-time hand tracking status (DETECTED / NO HAND)
  * Visual flash/notification confirming each successful sample capture.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Suppress low-level MediaPipe / TensorFlow C++ logging in terminal
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np

# Import hand detector and landmark extraction pipeline
try:
    from vision import SignLanguageHandDetector, extract_landmarks, extract_dual_landmarks, MODEL_PATH, ensure_model_downloaded
except ImportError:
    # Graceful fallback in case vision.py is not in current working directory
    import urllib.request
    import mediapipe as mp

    MODEL_FILENAME = "hand_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def ensure_model_downloaded(model_path: str = MODEL_PATH) -> str:
        if not os.path.exists(model_path):
            print(f"[*] Downloading MediaPipe model to '{model_path}'...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("[*] Model download complete.")
        return model_path

    def extract_landmarks(hand_landmarks) -> np.ndarray:
        if hand_landmarks is None:
            return np.zeros(21 * 3, dtype=np.float32)
        landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
        coords = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        if len(coords) != 63:
            features = np.zeros(63, dtype=np.float32)
            features[:min(len(coords), 63)] = coords[:63]
            return features
        return np.array(coords, dtype=np.float32)

    def extract_dual_landmarks(hand_landmarks_list, max_hands: int = 2) -> np.ndarray:
        total_features = max_hands * 21 * 3
        features = np.zeros(total_features, dtype=np.float32)
        if not hand_landmarks_list:
            return features
        for h_idx in range(min(len(hand_landmarks_list), max_hands)):
            hand = hand_landmarks_list[h_idx]
            single_feats = extract_landmarks(hand)
            features[h_idx * 63 : (h_idx + 1) * 63] = single_feats
        return features

    class SignLanguageHandDetector:
        def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
            self.max_num_hands = max_num_hands
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            self.use_tasks_api = not (hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"))
            self._timestamp_ms = 0

            if self.use_tasks_api:
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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.use_tasks_api:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self._timestamp_ms += 33
                result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
                return result.hand_landmarks
            else:
                frame_rgb.flags.writeable = False
                results = self.detector.process(frame_rgb)
                frame_rgb.flags.writeable = True
                return results.multi_hand_landmarks if results.multi_hand_landmarks else []

        def draw_landmarks_on_frame(self, frame_bgr: np.ndarray, hand_landmarks_list) -> None:
            if not hand_landmarks_list:
                return
            for hand_landmarks in hand_landmarks_list:
                if self.use_tasks_api:
                    self.drawing_utils.draw_landmarks(frame_bgr, hand_landmarks, self.connections)
                else:
                    if self.mp_drawing_styles:
                        landmark_style = self.mp_drawing_styles.get_default_hand_landmarks_style()
                        connection_style = self.mp_drawing_styles.get_default_hand_connections_style()
                    else:
                        landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        connection_style = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    self.mp_drawing.draw_landmarks(
                        frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, landmark_style, connection_style
                    )

        def close(self):
            if hasattr(self.detector, "close"):
                self.detector.close()


def generate_header(num_points: int = 42) -> List[str]:
    """Generates CSV header for either 42 points (126 coords) or 21 points (63 coords)."""
    header = ["label"]
    for i in range(num_points):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def init_csv_file(csv_path: str, target_points: int = 42) -> Tuple[int, int]:
    """
    Initializes the dataset CSV file with headers if it doesn't already exist.
    Returns (existing_sample_count, expected_feature_count).
    """
    existing_samples = 0
    expected_feats = target_points * 3
    file_exists = os.path.isfile(csv_path)

    if file_exists and os.path.getsize(csv_path) > 0:
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 0:
                    header = rows[0]
                    # If first col is 'label', data cols is len(header) - 1
                    if header and header[0].lower() == "label":
                        expected_feats = len(header) - 1
                        existing_samples = max(0, len(rows) - 1)
                    else:
                        existing_samples = len(rows)
        except Exception as e:
            print(f"[!] Warning reading existing CSV '{csv_path}': {e}")
    else:
        parent_dir = os.path.dirname(csv_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(generate_header(target_points))
        print(f"[*] Initialized new dataset file with {target_points}-point ({expected_feats} features) header: '{csv_path}'")

    return existing_samples, expected_feats


def append_sample_to_csv(csv_path: str, label: str, landmarks: List[float], expected_len: int = 126) -> bool:
    """
    Appends a new sample row to the CSV file.
    Row structure: [label, x0, y0, z0, ..., xN, yN, zN].
    """
    if len(landmarks) != expected_len:
        print(f"[!] Error: Expected {expected_len} landmark coordinates, got {len(landmarks)}.")
        return False

    formatted_landmarks = [round(float(v), 6) for v in landmarks]
    row = [label] + formatted_landmarks

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"[!] Error writing sample to '{csv_path}': {e}")
        return False


# Comprehensive Training Vocabulary for Citizen Services & Everyday Communication
DEFAULT_TRAINING_WORDS: List[str] = [
    # Civic & Greetings
    "HELLO", "THANK_YOU", "PLEASE", "YES", "NO", "WELCOME", "SORRY",
    # Urgent & Assistance
    "HELP", "EMERGENCY", "DOCTOR", "HOSPITAL", "MEDICINE", "PAIN",
    # Daily Essentials
    "WATER", "FOOD", "MORE", "DONE", "STOP",
    # Interaction & Society
    "FRIEND", "FAMILY", "TOGETHER", "LOVE", "PEACE", "WAIT", "TIME",
    "MONEY", "WHERE", "REPEAT", "GOOD", "BAD",
    # Pronouns & Feedback
    "I", "WE", "LIKE"
]


def draw_hud(
    frame: np.ndarray,
    session_count: int,
    total_file_samples: int,
    active_label: str,
    word_index: int,
    total_words: int,
    target_samples: int,
    hand_detected: bool,
    label_counts: Dict[str, int],
    feedback_msg: str,
    feedback_is_error: bool,
    feedback_timer: float,
) -> None:
    """
    Draws a modern, semi-transparent HUD overlay on the live feed showing:
    - Session sample counter & active word progress
    - Hand detection indicator (42 points / 2 hands)
    - Active word with index in training queue (e.g., [4/32] WATER: 18/25 samples)
    - Interactive hotkeys: TAB/n (Next Word), p (Prev Word), SPACE (Capture), b (Burst)
    - Feedback / flash notifications
    """
    h, w, _ = frame.shape

    # 1. Top HUD Banner (Semi-transparent dark overlay)
    banner_height = 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # 2. Session Samples Counter
    cv2.putText(
        frame,
        f"SESSION SAMPLES: {session_count}",
        (18, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 255, 128),  # Vibrant Emerald Green
        2,
        cv2.LINE_AA,
    )

    # 3. All-time / File sample count badge
    all_time_text = f"(Total in file: {total_file_samples + session_count})"
    cv2.putText(
        frame,
        all_time_text,
        (380, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # 4. Hand Detection Status Badge
    if hand_detected:
        status_text = "[HAND DETECTED - READY]"
        status_color = (0, 230, 100)  # Bright Green
    else:
        status_text = "[NO HAND DETECTED - POSITION HANDS]"
        status_color = (0, 120, 255)  # Orange-Red Alert

    cv2.putText(
        frame,
        status_text,
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # 5. Active Word & Target Progress Tracker
    active_clean = active_label.split()[0].upper()
    word_samples = label_counts.get(active_clean, 0)
    progress_pct = min(100, int((word_samples / max(1, target_samples)) * 100))

    word_badge = f"WORD [{word_index + 1}/{total_words}]: '{active_label}'"
    progress_badge = f"Progress: {word_samples}/{target_samples} ({progress_pct}%)"
    cv2.putText(
        frame,
        f"{word_badge}  |  {progress_badge}",
        (18, 92),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 215, 0),  # Amber Gold
        1,
        cv2.LINE_AA,
    )

    # Mini progress bar line
    bar_width = int((w - 36) * (progress_pct / 100.0))
    cv2.rectangle(frame, (18, 102), (18 + bar_width, 106), (0, 255, 128), -1)
    cv2.rectangle(frame, (18, 102), (w - 18, 106), (70, 70, 70), 1)

    # 6. Bottom Banner: Controls / Keybindings
    footer_height = 36
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - footer_height), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(footer_overlay, 0.78, frame, 0.22, 0, frame)

    controls_text = "SPACE: Record | b: Burst (x5) | TAB/n: Next Word | p: Prev Word | 'q': Exit"
    cv2.putText(
        frame,
        controls_text,
        (14, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (160, 205, 255),
        1,
        cv2.LINE_AA,
    )

    # 7. Action / Notification Banner (Flashes when sample is recorded or error occurs)
    now = time.time()
    if feedback_msg and (now - feedback_timer < 1.4):
        msg_color = (0, 60, 255) if feedback_is_error else (0, 255, 128)
        bg_color = (40, 20, 20) if feedback_is_error else (20, 45, 20)

        # Draw floating notification box
        box_y1 = banner_height + 10
        box_y2 = box_y1 + 38
        cv2.rectangle(frame, (14, box_y1), (w - 14, box_y2), bg_color, -1)
        cv2.rectangle(frame, (14, box_y1), (w - 14, box_y2), msg_color, 2)
        cv2.putText(
            frame,
            feedback_msg,
            (24, box_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            msg_color,
            2,
            cv2.LINE_AA,
        )

        # Border flash effect on capture
        if not feedback_is_error:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 128), 3)


def run_data_collection(
    csv_path: str = "landmarks.csv",
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    default_label: str = "HELLO",
    max_hands: int = 2,
    words_list: Optional[List[str]] = None,
    target_samples: int = 25,
) -> None:
    """
    Main webcam data collection loop.
    Captures frames, extracts 42 3D landmarks (126 floats for 2 hands or 63 for 1 hand),
    supports interactive cycling through vocabulary words, and logs samples to CSV.
    """
    window_name = "Sign Language Dataset Collector - collect_data.py"

    # Setup training vocabulary words
    if not words_list:
        words_list = list(DEFAULT_TRAINING_WORDS)

    default_upper = default_label.upper()
    if default_upper in words_list:
        word_index = words_list.index(default_upper)
    else:
        words_list.insert(0, default_upper)
        word_index = 0

    active_label = words_list[word_index]

    # Initialize CSV file and read existing sample count and feature dimension
    total_file_samples, expected_feats = init_csv_file(csv_path, target_points=max_hands * 21)

    # Optimize camera backend for Windows (CAP_DSHOW provides instant startup & low latency)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera with index {camera_index}.")
        print("[!] Please check webcam connection or try a different index (--camera 1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 65)
    print("  SIGN LANGUAGE VOCABULARY DATASET COLLECTOR INITIALIZED")
    print("=" * 65)
    print(f"- Output CSV File        : {os.path.abspath(csv_path)}")
    print(f"- Existing File Samples  : {total_file_samples}")
    print(f"- Expected Features      : {expected_feats} ({expected_feats // 3} points)")
    print(f"- Camera Resolution      : {actual_w}x{actual_h}")
    print(f"- Max Hands Tracked      : {max_hands} (Up to 42 keypoints)")
    print(f"- Total Training Words   : {len(words_list)} words")
    print(f"- Initial Active Word    : '{active_label}' (Index {word_index + 1}/{len(words_list)})")
    print(f"- Target Samples / Word  : {target_samples}")
    print("- Keybindings:")
    print("    * Press SPACEBAR           : Capture 1 sample for current word")
    print("    * Press 'b'                : Burst-record 5 samples in quick succession")
    print("    * Press TAB or 'n'         : Advance to NEXT word in vocabulary")
    print("    * Press 'p'                : Go to PREVIOUS word in vocabulary")
    print("    * Press '1'-'9'            : Jump directly to words 1 through 9")
    print("    * Press 'q' or ESC         : Save dataset and exit cleanly")
    print("=" * 65 + "\n")

    detector = SignLanguageHandDetector(max_num_hands=max_hands)

    session_samples = 0
    label_counts: Dict[str, int] = {}

    feedback_msg = ""
    feedback_is_error = False
    feedback_timer = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame for hand landmarks
            hands_landmarks = detector.process_frame(frame)
            hand_detected = bool(hands_landmarks and len(hands_landmarks) > 0)
            hands_count = len(hands_landmarks) if hands_landmarks else 0
            points_tracked = min(hands_count, 2) * 21

            # Draw landmarks on frame
            if hand_detected:
                detector.draw_landmarks_on_frame(frame, hands_landmarks)

            # Render HUD overlay on video frame
            draw_hud(
                frame=frame,
                session_count=session_samples,
                total_file_samples=total_file_samples,
                active_label=f"{active_label} ({points_tracked}/42 pts)",
                word_index=word_index,
                total_words=len(words_list),
                target_samples=target_samples,
                hand_detected=hand_detected,
                label_counts=label_counts,
                feedback_msg=feedback_msg,
                feedback_is_error=feedback_is_error,
                feedback_timer=feedback_timer,
            )

            cv2.imshow(window_name, frame)

            # Listen for keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Exit on 'q', 'Q', or ESC
            if key in (ord("q"), ord("Q"), 27):
                break

            # Handle window close button ('X')
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # 1. TAB (9) or 'n' / 'N': Next word
            if key in (9, ord("n"), ord("N")):
                word_index = (word_index + 1) % len(words_list)
                active_label = words_list[word_index]
                feedback_msg = f"[*] Switched to Word [{word_index + 1}/{len(words_list)}]: '{active_label}'"
                feedback_is_error = False
                feedback_timer = time.time()
                continue

            # 2. 'p' / 'P': Previous word
            elif key in (ord("p"), ord("P")):
                word_index = (word_index - 1) % len(words_list)
                active_label = words_list[word_index]
                feedback_msg = f"[*] Switched to Word [{word_index + 1}/{len(words_list)}]: '{active_label}'"
                feedback_is_error = False
                feedback_timer = time.time()
                continue

            # 3. 'b' / 'B': Burst-record 5 samples for active word
            elif key in (ord("b"), ord("B")):
                if not hand_detected:
                    feedback_msg = f"[!] No hand detected for burst capture of '{active_label}'!"
                    feedback_is_error = True
                    feedback_timer = time.time()
                else:
                    burst_success = 0
                    for _ in range(5):
                        if expected_feats == 126 or max_hands >= 2:
                            feats = extract_dual_landmarks(hands_landmarks, max_hands=2)
                        else:
                            feats = extract_landmarks(hands_landmarks[0])
                        jitter = np.random.normal(0, 0.003, size=feats.shape).astype(np.float32)
                        sample_row = [float(v) for v in (feats + jitter).tolist()]
                        if append_sample_to_csv(csv_path, active_label, sample_row, expected_len=expected_feats):
                            session_samples += 1
                            label_counts[active_label] = label_counts.get(active_label, 0) + 1
                            burst_success += 1
                        time.sleep(0.03)
                    feedback_msg = f"[+] Burst recorded +{burst_success} samples for '{active_label}'"
                    feedback_is_error = False
                    feedback_timer = time.time()
                continue

            # 4. Jump to word index using numbers '1' - '9'
            elif ord("1") <= key <= ord("9"):
                target_idx = key - ord("1")
                if target_idx < len(words_list):
                    word_index = target_idx
                    active_label = words_list[word_index]
                    feedback_msg = f"[*] Jumped to Word [{word_index + 1}/{len(words_list)}]: '{active_label}'"
                    feedback_is_error = False
                    feedback_timer = time.time()
                continue

            # 5. SPACEBAR: Record 1 sample for current active word
            target_label: Optional[str] = None
            if key == ord(" "):
                target_label = active_label

            if target_label is not None:
                if not hand_detected:
                    feedback_msg = f"[!] No hand detected! Position hand in frame to record '{target_label}'."
                    feedback_is_error = True
                    feedback_timer = time.time()
                    print(f"[*] Attempted to record '{target_label}', but no hand was detected.")
                else:
                    if expected_feats == 126 or max_hands >= 2:
                        features_1d = extract_dual_landmarks(hands_landmarks, max_hands=2)
                    else:
                        features_1d = extract_landmarks(hands_landmarks[0])

                    features_list = [float(v) for v in features_1d.tolist()]

                    # Append row to landmarks.csv
                    success = append_sample_to_csv(csv_path, target_label, features_list, expected_len=expected_feats)

                    if success:
                        session_samples += 1
                        label_counts[target_label] = label_counts.get(target_label, 0) + 1
                        feedback_msg = f"[+] Saved sample #{session_samples} for '{target_label}' ({label_counts[target_label]}/{target_samples})"
                        feedback_is_error = False
                        feedback_timer = time.time()
                        print(
                            f"[+] Recorded sample #{session_samples}: Label='{target_label}' "
                            f"(Count: {label_counts[target_label]}/{target_samples}, "
                            f"Hands: {hands_count}, Feats: {expected_feats})"
                        )
                    else:
                        feedback_msg = f"[!] Failed to write sample to '{csv_path}'."
                        feedback_is_error = True
                        feedback_timer = time.time()

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        print("\n" + "=" * 65)
        print("  SESSION SUMMARY")
        print("=" * 65)
        print(f"- Total Samples Collected This Session : {session_samples}")
        print(f"- Total Samples in Dataset File        : {total_file_samples + session_samples}")
        if label_counts:
            print("- Per-word sample breakdown:")
            for lbl, cnt in sorted(label_counts.items()):
                print(f"    * '{lbl}': {cnt} samples")
        print(f"- Saved to: {os.path.abspath(csv_path)}")
        print("=" * 65)

        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Resources released cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset collection tool for sign language vocabulary classification with OpenCV and MediaPipe Hands."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="landmarks.csv",
        help="Path to output CSV file (default: landmarks.csv)",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="HELLO",
        help="Initial active word label (e.g. HELLO, WATER, HELP; default: HELLO)",
    )
    parser.add_argument(
        "-w",
        "--words",
        type=str,
        default="",
        help="Comma-separated custom training words list (e.g. 'HELLO,WATER,FOOD,HELP'). Defaults to full vocabulary.",
    )
    parser.add_argument(
        "-t",
        "--target-samples",
        type=int,
        default=25,
        help="Target sample count per word to guide data collection (default: 25)",
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Webcam frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Webcam frame height (default: 480)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum hands to detect (default: 2, 42 points)",
    )

    args = parser.parse_args()

    words_list = None
    if args.words.strip():
        words_list = [w.strip().upper() for w in args.words.split(",") if w.strip()]

    run_data_collection(
        csv_path=args.output,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        default_label=args.label,
        max_hands=args.max_hands,
        words_list=words_list,
        target_samples=args.target_samples,
    )


if __name__ == "__main__":
    main()

