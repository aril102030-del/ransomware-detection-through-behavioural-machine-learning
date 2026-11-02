"""
detect_real.py
--------------
Use the trained real-data models to classify a process as:
  - Binary:     Benign  OR  Ransomware
  - Multiclass: Conti / CryptoLocker / LockBit / Normal / Ryuk / Sodinokibi / WannaCry

Usage:
  python detect_real.py --binary     # predict using binary model
  python detect_real.py --family     # predict ransomware family
  python detect_real.py --csv path\to\file.csv   # batch predict a CSV file
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(__file__)

def load_model(model_type="binary"):
    fname = "model_binary_real.pkl" if model_type == "binary" else "model_multiclass_real.pkl"
    path  = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun train_real.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_single(bundle, sample: dict):
    features = bundle["features"]
    X = np.array([[sample.get(f, 0) for f in features]], dtype=float)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]

    if bundle["type"] == "binary":
        label = "RANSOMWARE" if pred == 1 else "BENIGN"
        confidence = proba[pred]
        return label, confidence, proba
    else:
        le    = bundle["label_encoder"]
        label = le.inverse_transform([pred])[0]
        confidence = proba[pred]
        classes = bundle["classes"]
        return label, confidence, dict(zip(classes, proba))


def print_result_binary(label, confidence, proba):
    icon = "[RANSOMWARE]" if label == "RANSOMWARE" else "[BENIGN]"
    print(f"\n  +-- VERDICT: {icon}")
    print(f"  |  Confidence: {confidence:.2%}")
    print(f"  |  P(Benign):     {proba[0]:.4f}")
    print(f"  |  P(Ransomware): {proba[1]:.4f}")
    print("  +--------------------------------------------------")


def print_result_family(label, confidence, proba_dict):
    icon = "[BENIGN]" if label == "Normal" else "[RANSOMWARE]"
    print(f"\n  +-- VERDICT: {icon} Family = {label.upper()}")
    print(f"  |  Confidence: {confidence:.2%}")
    print(f"  |  Probability per class:")
    for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"  |    {cls:<15} {p:.4f}  {bar}")
    print("  +--------------------------------------------------")


def interactive_mode(bundle):
    features = bundle["features"]
    print(f"\n  Enter API call counts (press Enter to use 0 as default).")
    print(f"  There are {len(features)} features. Type 'skip' to fill remaining with 0.\n")
    sample = {}
    for feat in features:
        val = input(f"  {feat}: ").strip()
        if val.lower() == "skip":
            break
        sample[feat] = float(val) if val else 0.0

    label, conf, extra = predict_single(bundle, sample)
    if bundle["type"] == "binary":
        print_result_binary(label, conf, extra)
    else:
        print_result_family(label, conf, extra)


def batch_predict(bundle, csv_path):
    """Predict all rows in a CSV file and save results."""
    df = pd.read_csv(csv_path)
    features = bundle["features"]
    model = bundle["model"]

    # Fill missing feature columns with 0
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].fillna(0).values
    preds = model.predict(X)
    probas = model.predict_proba(X)

    if bundle["type"] == "binary":
        df["prediction"] = ["RANSOMWARE" if p == 1 else "BENIGN" for p in preds]
        df["confidence"]  = probas.max(axis=1)
        df["p_benign"]    = probas[:, 0]
        df["p_ransomware"] = probas[:, 1]
    else:
        le = bundle["label_encoder"]
        df["prediction"] = le.inverse_transform(preds)
        df["confidence"]  = probas.max(axis=1)
        for i, cls in enumerate(bundle["classes"]):
            df[f"p_{cls}"] = probas[:, i]

    out_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Processed {len(df)} rows.")
    print(f"  Results saved -> {out_path}")
    print(f"\n  Summary:")
    print(df["prediction"].value_counts().to_string())


def run_demo(bundle):
    """Demo with a clear ransomware-like and benign-like API profile."""
    demo_samples = {
        "Ransomware-like (high crypto + file ops)": {
            "NtCreateFile": 500, "NtWriteFile": 400, "NtReadFile": 300,
            "CryptEncrypt": 120, "CryptCreateHash": 80, "CryptGenKey": 50,
            "NtAllocateVirtualMemory": 200, "NtClose": 800,
            "DeleteFileW": 50, "MoveFileWithProgressW": 100,
            "SetFileAttributesW": 60, "NtOpenKey": 30,
        },
        "Benign-like (normal office app)": {
            "NtClose": 90, "NtOpenKey": 20, "NtQueryValueKey": 15,
            "LdrLoadDll": 10, "NtCreateFile": 8, "NtReadFile": 5,
            "RegOpenKeyExW": 6, "RegCloseKey": 4, "NtMapViewOfSection": 3,
        },
    }
    for name, sample in demo_samples.items():
        print(f"\n  Sample: {name}")
        label, conf, extra = predict_single(bundle, sample)
        if bundle["type"] == "binary":
            print_result_binary(label, conf, extra)
        else:
            print_result_family(label, conf, extra)


def main():
    parser = argparse.ArgumentParser(description="Ransomware Detector (Real Model)")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--binary", action="store_true", help="Use binary model (Benign vs Ransomware)")
    group.add_argument("--family", action="store_true", help="Use family model (identify ransomware strain)")
    parser.add_argument("--csv",  type=str, help="Path to CSV for batch prediction")
    parser.add_argument("--demo", action="store_true", help="Run demo samples")
    args = parser.parse_args()

    model_type = "multiclass" if args.family else "binary"
    bundle = load_model(model_type)
    print(f"\n  Model loaded: {'Binary (Benign vs Ransomware)' if model_type == 'binary' else 'Family Classifier (7 classes)'}")
    print(f"  Features: {len(bundle['features'])} Windows API call counts")

    if args.demo:
        run_demo(bundle)
    elif args.csv:
        batch_predict(bundle, args.csv)
    else:
        interactive_mode(bundle)


if __name__ == "__main__":
    main()
