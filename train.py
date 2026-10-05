"""
train.py - Sign Language Gesture Classifier Training Script

Trains a Support Vector Machine (SVM) classifier on hand gesture landmark coordinates
extracted by MediaPipe and saved in `landmarks.csv`.

Pipeline:
1. Loads `landmarks.csv` using pandas.
2. Extracts target labels (first column) and 63 3D landmark features (remaining columns).
3. Encodes string labels into integer categories using scikit-learn's LabelEncoder.
4. Splits dataset into 80% training and 20% testing sets using train_test_split.
5. Trains a Support Vector Classifier (SVC) with RBF or Linear kernel.
6. Evaluates test set accuracy and generates classification metrics.
7. Saves the trained model and label encoder to disk as `model.pkl` using joblib.
"""

import argparse
import os
import sys
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Loads landmark data from CSV using pandas.
    - First column: string target label (y).
    - Remaining 63 columns: landmark features (X).

    Returns:
        X: NumPy array of shape (N, 63) containing landmark features.
        y: 1D NumPy array of encoded integer labels.
        label_encoder: Fitted LabelEncoder instance.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file '{csv_path}' not found.\n"
            f"Please run 'python collect_data.py' first to collect gesture samples."
        )

    # Load CSV using pandas
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Dataset file '{csv_path}' is empty.")

    # First column is the target label (y)
    y_raw = df.iloc[:, 0].astype(str).values

    # Remaining columns are the landmark features (X)
    X = df.iloc[:, 1:].values.astype(np.float32)

    num_features = X.shape[1]
    if num_features not in (63, 126) and num_features % 3 != 0:
        raise ValueError(
            f"Expected 63 feature columns (21 keypoints) or 126 feature columns (42 keypoints) in '{csv_path}', but found {num_features}.\n"
            f"Each hand landmark sample must contain 3D coordinates (x, y, z)."
        )

    points_per_sample = num_features // 3
    print(f"[*] Dataset feature dimension: {num_features} columns ({points_per_sample} 3D landmarks)")

    # Encode string labels into numerical categories
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder


def train_svm(
    csv_path: str = "landmarks.csv",
    output_model_path: str = "model.pkl",
    kernel: str = "rbf",
    c_val: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Loads data, trains an SVM model, evaluates accuracy, and saves model.pkl.
    """
    print("=" * 65)
    print("       SIGN LANGUAGE GESTURE CLASSIFIER - SVM TRAINING")
    print("=" * 65)
    print(f"[*] Loading dataset from: {os.path.abspath(csv_path)}")

    X, y, label_encoder = load_dataset(csv_path)

    total_samples = len(y)
    classes = label_encoder.classes_
    num_classes = len(classes)

    print(f"[*] Total samples loaded: {total_samples}")
    print(f"[*] Total classes ({num_classes}): {list(classes)}")

    # Display per-class sample distribution
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        print(f"    - '{classes[cls_idx]}': {count} samples")

    if num_classes < 2:
        raise ValueError(
            f"Training requires at least 2 distinct classes, but only found {num_classes}: {list(classes)}.\n"
            f"Please run 'collect_data.py' to record samples for at least two different gestures."
        )

    # Stratified split if all classes have at least 2 samples, otherwise random split
    min_class_samples = min(counts)
    stratify = y if min_class_samples >= 2 else None
    if stratify is None:
        print("[!] Warning: Some classes have only 1 sample; stratified split disabled.")

    print(f"\n[*] Splitting dataset (Train: {int((1 - test_size) * 100)}%, Test: {int(test_size * 100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    print(f"    - Training set size : {len(X_train)} samples")
    print(f"    - Testing set size  : {len(X_test)} samples")

    # Initialize Support Vector Classifier (SVC)
    print(f"\n[*] Initializing Support Vector Classifier (Kernel='{kernel}', C={c_val})...")
    model = SVC(
        kernel=kernel,
        C=c_val,
        probability=True,  # Enables probability estimates for prediction confidence
        random_state=random_state,
    )

    # Train model
    print("[*] Training model on training set...")
    model.fit(X_train, y_train)
    print("[*] Training completed.")

    # Evaluate on testing set
    print("\n" + "=" * 65)
    print("                    EVALUATION RESULTS")
    print("=" * 65)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[*] Test Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    # Print Detailed Classification Report
    print("Classification Report:")
    target_names = [str(cls) for cls in classes]
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    print(report)

    # Save model and label encoder to disk
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "kernel": kernel,
        "accuracy": accuracy,
        "classes": list(classes),
    }

    output_dir = os.path.dirname(output_model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    joblib.dump(bundle, output_model_path)
    print(f"[*] Successfully saved model and label encoder to: {os.path.abspath(output_model_path)}")
    print("=" * 65)
    print("\nHow to load this model in your backend prediction script:")
    print("```python")
    print("import joblib")
    print(f"data = joblib.load('{output_model_path}')")
    print("model = data['model']")
    print("label_encoder = data['label_encoder']")
    print("prediction = model.predict(features_63)")
    print("gesture = label_encoder.inverse_transform(prediction)[0]")
    print("```\n")

    return bundle


def load_model(model_path: str = "model.pkl") -> Tuple[SVC, LabelEncoder]:
    """
    Helper function to load the trained model and label encoder from disk.
    Supports dictionary bundle {'model': ..., 'label_encoder': ...} or tuple (model, label_encoder).

    Returns:
        model: Trained SVC instance.
        label_encoder: Fitted LabelEncoder instance.
    """
    data = joblib.load(model_path)
    if isinstance(data, dict):
        return data["model"], data["label_encoder"]
    elif isinstance(data, (tuple, list)):
        return data[0], data[1]
    else:
        raise ValueError(f"Unrecognized model bundle format in '{model_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Train an SVM classifier on MediaPipe hand gesture landmarks (landmarks.csv)."
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="landmarks.csv",
        help="Path to landmark CSV dataset (default: landmarks.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.pkl",
        help="Path to save trained model bundle (default: model.pkl)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="SVM kernel type (default: rbf)",
    )
    parser.add_argument(
        "-C",
        "--c-val",
        type=float,
        default=1.0,
        help="SVM regularization parameter C (default: 1.0)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    train_svm(
        csv_path=args.data,
        output_model_path=args.output,
        kernel=args.kernel,
        c_val=args.c_val,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
