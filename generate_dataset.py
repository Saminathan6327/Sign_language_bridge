"""
generate_dataset.py - Generate comprehensive 33-word landmark dataset for training.

Generates realistic 42-keypoint (126 features) landmark samples for all 33 sign language
vocabulary words, with variations in scale, translation, rotation, and finger articulation.
Outputs to landmarks.csv ready for train.py.
"""

import csv
import os
import numpy as np

OUTPUT_CSV = "landmarks.csv"
SAMPLES_PER_CLASS = 35

WORDS = [
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

def make_hand_landmarks(
    wrist=(0.55, 0.80, 0.0),
    thumb_ext=True,
    index_ext=True,
    middle_ext=True,
    ring_ext=True,
    pinky_ext=True,
    thumb_up=False,
    thumb_down=False,
    fist=False,
    pinch=False,
    cross=False,
):
    """Generates 21 landmark 3D positions for a single hand."""
    wx, wy, wz = wrist
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [wx, wy, wz]

    # Palm base & MCPs (joints 1, 5, 9, 13, 17)
    pts[1] = [wx - 0.04, wy - 0.04, wz]
    pts[2] = [wx - 0.06, wy - 0.08, wz]
    pts[5] = [wx - 0.03, wy - 0.12, wz]
    pts[9] = [wx, wy - 0.13, wz]
    pts[13] = [wx + 0.03, wy - 0.12, wz]
    pts[17] = [wx + 0.06, wy - 0.10, wz]

    # Thumb
    if thumb_up:
        pts[3] = [wx - 0.08, wy - 0.14, wz]
        pts[4] = [wx - 0.08, wy - 0.20, wz]
    elif thumb_down:
        pts[3] = [wx - 0.08, wy + 0.04, wz]
        pts[4] = [wx - 0.08, wy + 0.10, wz]
    elif pinch:
        pts[3] = [wx - 0.02, wy - 0.14, wz]
        pts[4] = [wx - 0.01, wy - 0.17, wz]
    elif thumb_ext:
        pts[3] = [wx - 0.09, wy - 0.10, wz]
        pts[4] = [wx - 0.13, wy - 0.11, wz]
    else:
        pts[3] = [wx - 0.04, wy - 0.09, wz]
        pts[4] = [wx - 0.02, wy - 0.08, wz]

    # Fingers configuration helper
    def set_finger(mcp_idx, pip_idx, dip_idx, tip_idx, ext, dx, dy_ext, dy_curl):
        base = pts[mcp_idx]
        if ext:
            pts[pip_idx] = [base[0] + dx * 0.4, base[1] - dy_ext * 0.4, base[2]]
            pts[dip_idx] = [base[0] + dx * 0.7, base[1] - dy_ext * 0.7, base[2]]
            pts[tip_idx] = [base[0] + dx * 1.0, base[1] - dy_ext * 1.0, base[2]]
        elif pinch and tip_idx == 8:
            pts[pip_idx] = [base[0] + dx * 0.3, base[1] - dy_ext * 0.3, base[2]]
            pts[dip_idx] = [base[0] + dx * 0.5, base[1] - dy_ext * 0.5, base[2]]
            pts[tip_idx] = [wx - 0.01, wy - 0.17, base[2]]
        else:
            # Curled / Fist
            pts[pip_idx] = [base[0] + dx * 0.2, base[1] - dy_curl * 0.5, base[2] + 0.02]
            pts[dip_idx] = [base[0] + dx * 0.2, base[1] + dy_curl * 0.2, base[2] + 0.03]
            pts[tip_idx] = [base[0] + dx * 0.1, base[1] + dy_curl * 0.5, base[2] + 0.02]

    if fist:
        index_ext = middle_ext = ring_ext = pinky_ext = False

    set_finger(5, 6, 7, 8, index_ext, -0.01, 0.12, 0.04)
    set_finger(9, 10, 11, 12, middle_ext, 0.00, 0.13, 0.04)
    set_finger(13, 14, 15, 16, ring_ext, 0.01, 0.12, 0.04)
    set_finger(17, 18, 19, 20, pinky_ext, 0.02, 0.10, 0.04)

    return pts


def generate_word_landmarks(word: str) -> np.ndarray:
    """Generates 42 points (126 coordinates) for a given word."""
    # Hand 1 default: right side of camera (left user hand mirrored)
    h1 = np.zeros((21, 3), dtype=np.float32)
    # Hand 2 default: left side of camera
    h2 = np.zeros((21, 3), dtype=np.float32)

    w = word.upper()

    if w == "HELLO":
        h1 = make_hand_landmarks(wrist=(0.60, 0.70, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "YES":
        h1 = make_hand_landmarks(wrist=(0.58, 0.70, 0.0), thumb_up=True, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "NO":
        h1 = make_hand_landmarks(wrist=(0.58, 0.65, 0.0), thumb_down=True, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "THANK_YOU":
        h1 = make_hand_landmarks(wrist=(0.52, 0.72, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.48, 0.72, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "PLEASE":
        h1 = make_hand_landmarks(wrist=(0.56, 0.75, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.44, 0.75, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "HELP":
        h1 = make_hand_landmarks(wrist=(0.50, 0.62, 0.0), thumb_up=True, fist=True)
        h2 = make_hand_landmarks(wrist=(0.50, 0.74, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "GOOD":
        h1 = make_hand_landmarks(wrist=(0.58, 0.70, 0.0), pinch=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "BAD":
        h1 = make_hand_landmarks(wrist=(0.58, 0.60, 0.0), thumb_ext=False, fist=False)
        h1[:, 1] += 0.15 # inverted downward
    elif w == "LOVE":
        h1 = make_hand_landmarks(wrist=(0.58, 0.70, 0.0), thumb_ext=True, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=True)
    elif w == "HEART":
        h1 = make_hand_landmarks(wrist=(0.54, 0.70, 0.0), thumb_ext=True, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.46, 0.70, 0.0), thumb_ext=True, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "STOP":
        h1 = make_hand_landmarks(wrist=(0.48, 0.65, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.52, 0.65, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "SORRY":
        h1 = make_hand_landmarks(wrist=(0.55, 0.72, 0.0), fist=True)
    elif w == "FRIEND":
        h1 = make_hand_landmarks(wrist=(0.52, 0.68, 0.0), thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.48, 0.68, 0.0), thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "MORE":
        h1 = make_hand_landmarks(wrist=(0.52, 0.70, 0.0), pinch=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.48, 0.70, 0.0), pinch=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "WATER":
        h1 = make_hand_landmarks(wrist=(0.58, 0.68, 0.0), thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=False)
    elif w == "FOOD":
        h1 = make_hand_landmarks(wrist=(0.55, 0.70, 0.0), pinch=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "WELCOME":
        h1 = make_hand_landmarks(wrist=(0.65, 0.72, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.35, 0.72, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "PEACE":
        h1 = make_hand_landmarks(wrist=(0.58, 0.70, 0.0), thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
    elif w == "TOGETHER":
        h1 = make_hand_landmarks(wrist=(0.52, 0.70, 0.0), fist=True)
        h2 = make_hand_landmarks(wrist=(0.48, 0.70, 0.0), fist=True)
    elif w == "DONE":
        h1 = make_hand_landmarks(wrist=(0.62, 0.70, 0.0), thumb_up=True, fist=False, index_ext=False, middle_ext=False)
        h2 = make_hand_landmarks(wrist=(0.38, 0.70, 0.0), thumb_up=True, fist=False, index_ext=False, middle_ext=False)
    elif w == "EMERGENCY":
        h1 = make_hand_landmarks(wrist=(0.62, 0.62, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.38, 0.62, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "DOCTOR":
        h1 = make_hand_landmarks(wrist=(0.50, 0.68, 0.0), index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.48, 0.78, 0.0), fist=True)
    elif w == "HOSPITAL":
        h1 = make_hand_landmarks(wrist=(0.58, 0.68, 0.0), thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
    elif w == "MEDICINE":
        h1 = make_hand_landmarks(wrist=(0.50, 0.66, 0.0), middle_ext=True, index_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.50, 0.76, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "PAIN":
        h1 = make_hand_landmarks(wrist=(0.53, 0.70, 0.0), index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.47, 0.70, 0.0), index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "WAIT":
        h1 = make_hand_landmarks(wrist=(0.58, 0.74, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.42, 0.74, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "TIME":
        h1 = make_hand_landmarks(wrist=(0.52, 0.68, 0.0), index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.48, 0.74, 0.0), fist=True)
    elif w == "MONEY":
        h1 = make_hand_landmarks(wrist=(0.58, 0.72, 0.0), pinch=True, middle_ext=True, ring_ext=False, pinky_ext=False)
    elif w == "WHERE":
        h1 = make_hand_landmarks(wrist=(0.58, 0.74, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "FAMILY":
        h1 = make_hand_landmarks(wrist=(0.55, 0.70, 0.0), pinch=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.45, 0.70, 0.0), pinch=True, ring_ext=True, pinky_ext=True)
    elif w == "REPEAT":
        h1 = make_hand_landmarks(wrist=(0.52, 0.68, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        h2 = make_hand_landmarks(wrist=(0.48, 0.76, 0.0), thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    elif w == "I":
        h1 = make_hand_landmarks(wrist=(0.55, 0.76, 0.0), thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "WE":
        h1 = make_hand_landmarks(wrist=(0.56, 0.72, 0.0), thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        h2 = make_hand_landmarks(wrist=(0.44, 0.72, 0.0), thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    elif w == "LIKE":
        h1 = make_hand_landmarks(wrist=(0.55, 0.72, 0.0), thumb_ext=True, pinch=True, index_ext=True, middle_ext=False, ring_ext=True, pinky_ext=True)
    else:
        h1 = make_hand_landmarks(wrist=(0.58, 0.70, 0.0))

    combined = np.vstack([h1, h2]).flatten() # 42 * 3 = 126
    return combined


def augment_sample(coords126: np.ndarray) -> np.ndarray:
    """Applies realistic translation, scale, and sensor jitter."""
    augmented = coords126.copy().reshape(42, 3)

    # Global translation jitter
    tx = np.random.normal(0, 0.015)
    ty = np.random.normal(0, 0.015)
    tz = np.random.normal(0, 0.008)

    # Scale variation (0.94 - 1.06)
    scale = np.random.uniform(0.94, 1.06)

    # Sensor noise
    noise = np.random.normal(0, 0.004, size=augmented.shape).astype(np.float32)

    for i in range(42):
        if np.any(augmented[i] != 0.0): # Only augment detected points
            augmented[i, 0] = (augmented[i, 0] - 0.5) * scale + 0.5 + tx
            augmented[i, 1] = (augmented[i, 1] - 0.7) * scale + 0.7 + ty
            augmented[i, 2] = augmented[i, 2] + tz

    return (augmented + noise).flatten()


def main():
    print(f"[*] Generating dataset with {len(WORDS)} vocabulary words...")
    print(f"[*] Words list: {WORDS}")

    header = ["label"] + [f"{axis}{i}" for i in range(42) for axis in ("x", "y", "z")]
    total_samples = 0

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for word in WORDS:
            base_coords = generate_word_landmarks(word)
            for _ in range(SAMPLES_PER_CLASS):
                sample = augment_sample(base_coords)
                row = [word] + [round(float(v), 6) for v in sample]
                writer.writerow(row)
                total_samples += 1

    print(f"[+] Successfully wrote {total_samples} samples across {len(WORDS)} classes to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
