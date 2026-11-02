"""
train_real.py
-------------
Train a ransomware detection model on the REAL dataset provided.

Dataset structure discovered:
  - ransomset-multiclass-dataset.csv : 19,234 rows x 240 cols
      * 238 API call count features (Windows NT/Win32 API names)
      * 'score_binary' : 0=Benign, 1=Ransomware  (binary label)
      * 'PRF'          : process/feature info (dropped)
      * 'classe'       : family name (Sodinikibi, Normal, Conti, CryptoLocker, LockBit)
  - Individual family CSVs (semicolon-separated, single column with header = feature names)
      Conti.csv (14010), Normal.csv (2103), WannaCry.csv (1480),
      Ryuk.csv (802), LockBit.csv (300), CryptoLocker.csv (360), Sodinokibi.csv (179)

We train TWO models:
  1. Binary classifier  : Benign vs Ransomware  (using ransomset-multiclass-dataset.csv)
  2. Multiclass classifier: Benign / Conti / WannaCry / Ryuk / LockBit / CryptoLocker / Sodinokibi
     (using individual family CSVs which have consistent semicolon-separated features)
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUT_DIR     = os.path.dirname(__file__)
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Helper: load the big multiclass CSV ───────────────────────────────────────

def load_multiclass_csv():
    path = os.path.join(DATASET_DIR, "ransomset-multiclass-dataset.csv")
    df   = pd.read_csv(path)
    print(f"  Loaded ransomset-multiclass-dataset.csv: {df.shape}")

    # Drop non-feature columns
    drop_cols = [c for c in ["PRF", "classe", "score_binary"] if c in df.columns]
    label_col = "score_binary"   # 0=benign, 1=ransomware
    family_col = "classe"

    # 'score_binary' is a float score, not a 0/1 label — derive binary from 'classe'
    y_family = df[family_col].values if family_col in df.columns else None
    y_binary = (df[family_col] != "Normal").astype(int).values if family_col in df.columns else None
    X = df.drop(columns=drop_cols, errors="ignore")

    # Coerce everything to numeric, fill NaN with 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y_binary, y_family, X.columns.tolist()


# ── Helper: load individual family CSVs (semicolon-delimited in one column) ───

def load_family_csvs():
    family_files = {
        "Conti":        "Dataset - Conti.csv",
        "CryptoLocker": "Dataset - CryptoLocker.csv",
        "LockBit":      "Dataset - LockBit.csv",
        "Normal":       "Dataset - Normal.csv",
        "Ryuk":         "Dataset - Ryuk.csv",
        "Sodinokibi":   "Dataset - Sodinokibi.csv",
        "WannaCry":     "Dataset - WannaCry.csv",
    }
    frames = []
    for family, fname in family_files.items():
        path = os.path.join(DATASET_DIR, fname)
        raw  = pd.read_csv(path)
        # The header IS the semicolon-joined feature names; values are rows
        col_names = raw.columns[0].split(";")
        rows = raw.iloc[:, 0].str.split(";", expand=True)
        rows.columns = col_names
        rows = rows.apply(pd.to_numeric, errors="coerce").fillna(0)
        rows["family"] = family
        rows["label"]  = 0 if family == "Normal" else 1
        frames.append(rows)
        print(f"  {family}: {len(rows)} rows x {len(col_names)} features")

    df = pd.concat(frames, ignore_index=True)
    # Align columns — fill missing cols with 0
    feature_cols = [c for c in df.columns if c not in ["family", "label"]]
    return df, feature_cols


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels)-1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_feature_importance(importances, feature_names, top_n, title, path):
    fi = pd.Series(importances, index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_class_distribution(y, label_map, title, path):
    vals, counts = np.unique(y, return_counts=True)
    bar_labels = [label_map.get(int(v), str(v)) for v in vals]
    colors = ["#2ca02c" if "Normal" in str(label_map.get(int(v), v)) or int(v) == 0
              else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(max(5, len(vals)*1.5), 4))
    ax.bar(bar_labels, counts, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Sample count")
    for i, c in enumerate(counts):
        ax.text(i, c + 5, str(c), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 : Binary  (Benign vs Ransomware) using multiclass CSV
# ─────────────────────────────────────────────────────────────────────────────

def train_binary():
    print("\n" + "=" * 65)
    print("  MODEL 1 : Binary Ransomware Detector")
    print("=" * 65)

    X, y, y_family, feature_names = load_multiclass_csv()

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution: Benign={counts[0]}  Ransomware={counts[1]}")
    plot_class_distribution(
        y, {0: "Normal", 1: "Ransomware"},
        "Binary Class Distribution", os.path.join(PLOTS_DIR, "binary_class_dist.png"))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.25, random_state=42, stratify=y)

    print(f"  Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1")
    print(f"  5-Fold CV F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=["Benign", "Ransomware"]))
    print(f"  ROC-AUC : {roc_auc_score(y_te, y_proba):.4f}")

    # Train vs test gap
    train_f1 = f1_score(y_tr, model.predict(X_tr))
    test_f1  = f1_score(y_te, y_pred)
    gap = train_f1 - test_f1
    print(f"  Train F1={train_f1:.4f}  Test F1={test_f1:.4f}  Gap={gap:.4f}")
    print(f"  Fit diagnosis: {'OVERFITTING' if gap > 0.05 else 'GOOD FIT'}")

    # Plots
    cm = confusion_matrix(y_te, y_pred)
    plot_cm(cm, ["Benign", "Ransomware"], "Binary Confusion Matrix",
            os.path.join(PLOTS_DIR, "real_binary_cm.png"))
    plot_feature_importance(
        model.feature_importances_, feature_names, 20,
        "Top 20 API Features (Binary Model)",
        os.path.join(PLOTS_DIR, "real_binary_feature_importance.png"))

    # Save
    bundle = {"model": model, "features": feature_names,
              "type": "binary", "threshold": 0.5}
    with open(os.path.join(OUT_DIR, "model_binary_real.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    print("  Model saved -> model_binary_real.pkl")
    return model, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 : Multiclass  (Normal / Conti / WannaCry / Ryuk / LockBit / ...)
# ─────────────────────────────────────────────────────────────────────────────

def train_multiclass():
    print("\n" + "=" * 65)
    print("  MODEL 2 : Ransomware Family Classifier (Multiclass)")
    print("=" * 65)

    df, feature_cols = load_family_csvs()

    # Align all rows to the same feature set (fill missing with 0)
    X = df[feature_cols].fillna(0)
    y_raw = df["family"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    print(f"\n  Classes: {class_names}")

    # Distribution
    unique, counts = np.unique(y_raw, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u}: {c}")

    # Balance via upsampling minority classes to min 300
    parts = []
    for cls in class_names:
        sub = df[df["family"] == cls]
        if len(sub) < 300:
            sub = resample(sub, replace=True, n_samples=300, random_state=42)
        parts.append(sub)
    df_bal = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    X_bal = df_bal[feature_cols].fillna(0).values
    y_bal = le.transform(df_bal["family"].values)
    print(f"\n  Balanced dataset: {X_bal.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.25, random_state=42, stratify=y_bal)

    model = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")

    y_pred = model.predict(X_te)
    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names))

    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc  = accuracy_score(y_te, y_pred)
    gap = train_acc - test_acc
    print(f"  Train Acc={train_acc:.4f}  Test Acc={test_acc:.4f}  Gap={gap:.4f}")
    print(f"  Fit diagnosis: {'OVERFITTING' if gap > 0.05 else 'GOOD FIT'}")

    cm = confusion_matrix(y_te, y_pred)
    plot_cm(cm, class_names, "Ransomware Family Confusion Matrix",
            os.path.join(PLOTS_DIR, "real_multiclass_cm.png"))
    plot_feature_importance(
        model.feature_importances_, feature_cols, 20,
        "Top 20 API Features (Multiclass Model)",
        os.path.join(PLOTS_DIR, "real_multiclass_feature_importance.png"))

    bundle = {"model": model, "features": feature_cols,
              "label_encoder": le, "classes": class_names, "type": "multiclass"}
    with open(os.path.join(OUT_DIR, "model_multiclass_real.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    print("  Model saved -> model_multiclass_real.pkl")
    return model, feature_cols, le


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Real Dataset Training Pipeline")
    print("  Features: Windows API call counts (behavioural)")
    print("=" * 65)
    train_binary()
    train_multiclass()
    print("\n  All done! Check plots/ for confusion matrices and feature importance.")
