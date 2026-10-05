"""
predict.py - Standalone Inference Script for Hand Gesture Classification

Accepts a 63-coordinate floating-point array (21 3D landmarks: x, y, z),
loads the pre-trained scikit-learn SVM model (model.pkl),
and prints the predicted gesture/word.

Usage:
  1. Via stdin (recommended for child_process.spawn):
     echo [0.1, 0.2, ...] | python predict.py
  2. Via CLI argument:
     python predict.py "[0.1, 0.2, ...]"
"""

import json
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow / C++ logs if imported indirectly
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def load_model(model_path: str = MODEL_FILE):
    """
    Loads the trained SVM model and label encoder from disk.
    Supports dictionary bundle or tuple structure.
    """
    if not os.path.exists(model_path):
        return None, None

    try:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            model = bundle.get("model")
            label_encoder = bundle.get("label_encoder")
            return model, label_encoder
        elif isinstance(bundle, (tuple, list)):
            return bundle[0], bundle[1]
        else:
            return bundle, None
    except Exception as e:
        sys.stderr.write(f"[!] Error loading model from '{model_path}': {e}\n")
        return None, None


def parse_input_coordinates():
    """
    Reads coordinates from either command line argument or standard input.
    Expected format: JSON array of 63 numbers or JSON object with 'landmarks'/'features'.
    """
    raw_text = ""

    # Check CLI argument
    if len(sys.argv) > 1 and sys.argv[1].strip():
        raw_text = sys.argv[1].strip()
    else:
        # Read from stdin
        raw_text = sys.stdin.read().strip()

    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            coords = parsed.get("landmarks") or parsed.get("features") or parsed.get("coordinates")
            return coords
        elif isinstance(parsed, list):
            return parsed
        return None
    except Exception as e:
        sys.stderr.write(f"[!] JSON parse error: {e}\n")
        return None


def calculate_dual_hand_gesture(coords):
    """
    Heuristic rule-based real-time gesture recognizer for dual-hand (42 points = 126 floats)
    and single-hand (21 points = 63 floats).
    Returns (gesture_name, confidence) or None.
    """
    if len(coords) < 63:
        return None

    # Points helper
    def get_pt(idx):
        base = idx * 3
        return coords[base], coords[base + 1], coords[base + 2]

    def dist_2d(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # Check if Hand 2 is active (points 21 to 41)
    has_hand2 = False
    if len(coords) >= 126:
        h2_sum = sum(abs(coords[i]) for i in range(63, 126))
        if h2_sum > 0.01:
            has_hand2 = True

    # Helper: Finger states for a hand (offset 0 or 21)
    def get_finger_states(offset=0):
        wrist = get_pt(offset + 0)
        thumb_tip = get_pt(offset + 4)
        thumb_ip = get_pt(offset + 3)
        thumb_mcp = get_pt(offset + 2)

        index_tip = get_pt(offset + 8)
        index_pip = get_pt(offset + 6)
        index_mcp = get_pt(offset + 5)

        middle_tip = get_pt(offset + 12)
        middle_pip = get_pt(offset + 10)
        middle_mcp = get_pt(offset + 9)

        ring_tip = get_pt(offset + 16)
        ring_pip = get_pt(offset + 14)
        ring_mcp = get_pt(offset + 13)

        pinky_tip = get_pt(offset + 20)
        pinky_pip = get_pt(offset + 18)
        pinky_mcp = get_pt(offset + 17)

        # Orientation-invariant Euclidean ratios along kinematic finger chains
        index_ext = dist_2d(index_tip, wrist) > dist_2d(index_pip, wrist) * 1.12 and dist_2d(index_tip, index_mcp) > dist_2d(index_pip, index_mcp) * 0.95
        middle_ext = dist_2d(middle_tip, wrist) > dist_2d(middle_pip, wrist) * 1.12 and dist_2d(middle_tip, middle_mcp) > dist_2d(middle_pip, middle_mcp) * 0.95
        ring_ext = dist_2d(ring_tip, wrist) > dist_2d(ring_pip, wrist) * 1.12 and dist_2d(ring_tip, ring_mcp) > dist_2d(ring_pip, ring_mcp) * 0.95
        pinky_ext = dist_2d(pinky_tip, wrist) > dist_2d(pinky_pip, wrist) * 1.12 and dist_2d(pinky_tip, pinky_mcp) > dist_2d(pinky_pip, pinky_mcp) * 0.95

        # Thumb extension & directional orientation
        thumb_ext = dist_2d(thumb_tip, index_mcp) > 0.075 and dist_2d(thumb_tip, wrist) > dist_2d(thumb_mcp, wrist) * 1.12
        thumb_up = thumb_tip[1] < thumb_mcp[1] - 0.035
        thumb_down = thumb_tip[1] > thumb_mcp[1] + 0.035

        return {
            "wrist": wrist,
            "thumb_tip": thumb_tip,
            "thumb_mcp": thumb_mcp,
            "index_tip": index_tip,
            "index_pip": index_pip,
            "index_mcp": index_mcp,
            "middle_tip": middle_tip,
            "middle_pip": middle_pip,
            "ring_tip": ring_tip,
            "pinky_tip": pinky_tip,
            "pinky_mcp": pinky_mcp,
            "thumb_ext": thumb_ext,
            "thumb_up": thumb_up,
            "thumb_down": thumb_down,
            "index_ext": index_ext,
            "middle_ext": middle_ext,
            "ring_ext": ring_ext,
            "pinky_ext": pinky_ext,
        }

    h1 = get_finger_states(0)

    # 1. Dual-Hand Gestures (42 points)
    if has_hand2:
        h2 = get_finger_states(21)
        wrist_dist = dist_2d(h1["wrist"], h2["wrist"])
        index_tips_dist = dist_2d(h1["index_tip"], h2["index_tip"])
        thumb_tips_dist = dist_2d(h1["thumb_tip"], h2["thumb_tip"])
        middle_tips_dist = dist_2d(h1["middle_tip"], h2["middle_tip"])
        pinky_tips_dist = dist_2d(h1["pinky_tip"], h2["pinky_tip"])

        # 1. HEART SHAPE (Thumbs & index tips touching, pinkies tucked)
        if (
            h1["thumb_ext"]
            and h2["thumb_ext"]
            and thumb_tips_dist < 0.08
            and index_tips_dist < 0.08
            and not h1["pinky_ext"]
            and not h2["pinky_ext"]
            and wrist_dist < 0.34
        ):
            return "HEART", 0.98

        # 2. STOP / CROSSED WRISTS (X)
        if wrist_dist < 0.13 and index_tips_dist > 0.22:
            return "STOP", 0.96

        # 3. THANK YOU / NAMASTE (Both palms touching upright in prayer)
        all_h1_up = h1["index_ext"] and h1["middle_ext"] and h1["ring_ext"]
        all_h2_up = h2["index_ext"] and h2["middle_ext"] and h2["ring_ext"]
        if (
            all_h1_up
            and all_h2_up
            and wrist_dist < 0.28
            and index_tips_dist < 0.14
            and middle_tips_dist < 0.14
            and thumb_tips_dist < 0.16
        ):
            return "THANK_YOU", 0.98

        # 4. FRIEND (Both index fingers hooked / interlinked touching, thumbs separated)
        if (
            index_tips_dist < 0.085
            and thumb_tips_dist > 0.085
            and h1["index_ext"]
            and h2["index_ext"]
            and not h1["ring_ext"]
            and not h2["ring_ext"]
            and not h1["pinky_ext"]
            and not h2["pinky_ext"]
        ):
            return "FRIEND", 0.95

        # 4b. WE (Both index fingers pointing together side-by-side)
        if (
            index_tips_dist >= 0.075
            and index_tips_dist < 0.22
            and h1["index_ext"]
            and h2["index_ext"]
            and not h1["middle_ext"]
            and not h2["middle_ext"]
        ):
            return "WE", 0.96

        # 5. HELP (One fist resting on flat palm of other hand)
        h1_flat_h2_fist = all_h1_up and not h2["index_ext"] and not h2["middle_ext"]
        h2_flat_h1_fist = all_h2_up and not h1["index_ext"] and not h1["middle_ext"]
        if (h1_flat_h2_fist or h2_flat_h1_fist) and wrist_dist < 0.28:
            return "HELP", 0.95

        # 6. MORE (Both hands pinched, fingertips touching together)
        if (
            not h1["index_ext"]
            and not h2["index_ext"]
            and index_tips_dist < 0.09
            and thumb_tips_dist < 0.09
            and not h1["ring_ext"]
            and not h2["ring_ext"]
            and wrist_dist < 0.25
        ):
            return "MORE", 0.94

        # 7. TOGETHER (Both fists touching side by side)
        if (
            wrist_dist < 0.25
            and not h1["index_ext"]
            and not h1["middle_ext"]
            and not h2["index_ext"]
            and not h2["middle_ext"]
        ):
            return "TOGETHER", 0.94

        # 8. DONE (Dual thumbs up)
        h1_thumb_up = h1["thumb_ext"] and not h1["index_ext"] and not h1["middle_ext"] and not h1["ring_ext"] and not h1["pinky_ext"] and h1["thumb_tip"][1] < h1["thumb_mcp"][1]
        h2_thumb_up = h2["thumb_ext"] and not h2["index_ext"] and not h2["middle_ext"] and not h2["ring_ext"] and not h2["pinky_ext"] and h2["thumb_tip"][1] < h2["thumb_mcp"][1]
        if h1_thumb_up and h2_thumb_up:
            return "DONE", 0.98

        # 9. FOOD (Both open palms held side by side facing upward)
        if (
            pinky_tips_dist < 0.16
            and wrist_dist < 0.26
            and all_h1_up
            and all_h2_up
        ):
            return "FOOD", 0.94

        # 10. PLEASE (Both open cupped palms together gently)
        if all_h1_up and all_h2_up and wrist_dist < 0.25 and index_tips_dist < 0.20:
            return "PLEASE", 0.93

        # 11. WELCOME (Both open palms spread wide welcoming)
        if all_h1_up and all_h2_up and h1["pinky_ext"] and h2["pinky_ext"] and wrist_dist > 0.32:
            return "WELCOME", 0.95

    # ---------------------------------------------------------------------------
    # 2. Single-Hand Beginner Vocabulary (21 Points)
    # ---------------------------------------------------------------------------
    else:
        # 12. I LOVE YOU (ASL ILY - Thumb + Index + Pinky)
        if (
            h1["thumb_ext"]
            and h1["index_ext"]
            and not h1["middle_ext"]
            and not h1["ring_ext"]
            and h1["pinky_ext"]
        ):
            return "LOVE", 0.97

        # 13. GOOD (OK Sign - Thumb & Index circle)
        ok_dist = dist_2d(h1["thumb_tip"], h1["index_tip"])
        if ok_dist < 0.085 and h1["middle_ext"] and h1["ring_ext"]:
            return "GOOD", 0.96

        # 14. WATER (W sign - Index + Middle + Ring upright)
        if (
            h1["index_ext"]
            and h1["middle_ext"]
            and h1["ring_ext"]
            and not h1["pinky_ext"]
        ):
            return "WATER", 0.95

        # 15. PEACE (V sign - Index + Middle upright)
        if (
            h1["index_ext"]
            and h1["middle_ext"]
            and not h1["ring_ext"]
            and not h1["pinky_ext"]
        ):
            return "PEACE", 0.96

        # 16. YES (Thumb Up)
        if (
            h1["thumb_ext"]
            and not h1["index_ext"]
            and not h1["middle_ext"]
            and not h1["ring_ext"]
            and not h1["pinky_ext"]
            and h1["thumb_tip"][1] < h1["index_mcp"][1] - 0.035
        ):
            return "YES", 0.97

        # 17. NO (Thumb Down)
        if (
            h1["thumb_ext"]
            and not h1["index_ext"]
            and not h1["middle_ext"]
            and not h1["ring_ext"]
            and not h1["pinky_ext"]
            and h1["thumb_tip"][1] > h1["thumb_mcp"][1] + 0.04
        ):
            return "NO", 0.95

        # 18. HELLO (Open Palm Wave)
        if (
            h1["thumb_ext"]
            and h1["index_ext"]
            and h1["middle_ext"]
            and h1["ring_ext"]
            and h1["pinky_ext"]
        ):
            return "HELLO", 0.94

        # 19. BAD (Hand turned downward)
        if (
            h1["index_tip"][1] > h1["wrist"][1]
            and h1["middle_tip"][1] > h1["wrist"][1]
            and not h1["thumb_ext"]
        ):
            return "BAD", 0.92

        # 20. I / ME (Index finger pointing or pinky upright)
        if (h1["index_ext"] and not h1["middle_ext"] and not h1["ring_ext"] and not h1["pinky_ext"]) or (h1["pinky_ext"] and not h1["index_ext"] and not h1["middle_ext"]):
            return "I", 0.97

        # 21. LIKE (Thumb & Middle finger pinching in 8-handshape)
        if dist_2d(h1["thumb_tip"], h1["middle_tip"]) < 0.065 and h1["index_ext"] and h1["pinky_ext"]:
            return "LIKE", 0.97

        # 22. SORRY (Closed Fist over chest)
        if (
            not h1["index_ext"]
            and not h1["middle_ext"]
            and not h1["ring_ext"]
            and not h1["pinky_ext"]
            and not h1["thumb_up"]
        ):
            return "SORRY", 0.92

    return None


def main():
    coords = parse_input_coordinates()

    if not coords or len(coords) not in (63, 126):
        sys.stderr.write(f"[!] Invalid input: Expected 63 or 126 coordinates, got {len(coords) if coords else 0}\n")
        print(json.dumps({"prediction": "—", "confidence": 0.0, "error": "Invalid feature length"}))
        return

    # Check for heuristic gesture calculation first (high accuracy for rich vocabulary)
    detected_gesture = calculate_dual_hand_gesture(coords)

    # Convert coordinates for ML model
    features_all = np.array(coords, dtype=np.float32).reshape(1, -1)

    # Load model
    model, label_encoder = load_model(MODEL_FILE)

    predicted_word = None
    confidence = 0.0

    if model is not None:
        try:
            expected_feats = getattr(model, "n_features_in_", 63)
            if expected_feats == 63 and features_all.shape[1] == 126:
                # Slicing the primary hand 63 features for backwards compatibility
                features_for_model = features_all[:, :63]
            elif expected_feats == features_all.shape[1]:
                features_for_model = features_all
            else:
                features_for_model = None

            if features_for_model is not None:
                pred_idx = model.predict(features_for_model)
                if label_encoder is not None:
                    predicted_word = str(label_encoder.inverse_transform(pred_idx)[0])
                else:
                    predicted_word = str(pred_idx[0])

                confidence = 0.90
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(features_for_model)[0]
                        max_p = float(np.max(probabilities))
                        n_cls = len(probabilities)
                        chance = 1.0 / max(1, n_cls)
                        confidence = min(0.99, max(0.65, 0.50 + (max_p - chance) * 1.6))
                    except Exception:
                        pass
        except Exception as e:
            sys.stderr.write(f"[!] Model inference error: {e}\n")

    # If heuristic recognized a specific gesture or model gave no result
    if detected_gesture:
        heuristic_name, heuristic_conf = detected_gesture
        if not predicted_word:
            predicted_word = heuristic_name
            confidence = heuristic_conf
        elif heuristic_name == predicted_word:
            confidence = max(confidence, heuristic_conf)
        elif confidence < 0.50:
            predicted_word = heuristic_name
            confidence = heuristic_conf

    if not predicted_word:
        predicted_word = "READY"
        confidence = 0.50

    output = {
        "prediction": predicted_word,
        "confidence": round(confidence, 4),
        "handsDetected": 2 if (len(coords) == 126 and sum(abs(coords[i]) for i in range(63, 126)) > 0.01) else 1,
        "points": 42 if len(coords) == 126 else 21,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
