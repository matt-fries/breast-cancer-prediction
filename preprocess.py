from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_preprocessing(csv_path="TrainingPortion.csv", test_size=0.20, random_state=42):
    summary_lines = []

    def checkpoint(label, detail):
        msg = f"[QC] {label}: {detail}"
        print(msg)
        summary_lines.append(msg)

    # ── Load data ──
    df = pd.read_csv(csv_path)
    checkpoint("Load", f"{df.shape[0]} rows, {df.shape[1]} columns from {csv_path}")
    checkpoint("Columns", ", ".join(df.columns))
    checkpoint("Dtypes", str(df.dtypes.value_counts().to_dict()))

    # ── Drop id column ──
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
        checkpoint("Drop column", "Removed 'id' column")
    checkpoint("Shape after drop", f"{df.shape}")

    # ── Check for missing values ──
    missing = df.isnull().sum().sum()
    checkpoint("Missing values", f"{missing} total")

    # ── Check for duplicates ──
    duplicates = df.duplicated().sum()
    checkpoint("Duplicate rows", f"{duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        checkpoint("After dedup", f"{df.shape[0]} rows remaining")

    # ── Encode diagnosis ──
    before_counts = df["diagnosis"].value_counts().to_dict()
    checkpoint("Diagnosis before encoding", str(before_counts))

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    after_counts = df["diagnosis"].value_counts().to_dict()
    checkpoint("Diagnosis after encoding", str(after_counts))

    mal_pct = df["diagnosis"].mean()
    checkpoint("Class balance", f"{mal_pct:.1%} malignant, {1 - mal_pct:.1%} benign")

    # ── Drop redundant high-correlation features ──
    drop_cols = [
        "perimeter_mean",
        "area_mean",  # redundant with radius_mean (r > 0.98)
        "perimeter_se",
        "area_se",  # redundant with radius_se (r > 0.95)
        "perimeter_worst",
        "area_worst",  # redundant with radius_worst (r > 0.98)
        "concave points_mean",  # redundant with concavity_mean (r = 0.92)
        "texture_mean",  # redundant with texture_worst (r = 0.91)
        "concave points_worst",  # redundant with concavity_mean (r = 0.91)
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    checkpoint(
        "Dropped redundant features",
        f"{len(existing_drops)} columns: {', '.join(existing_drops)}",
    )

    # ── Split features and target ──
    x = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    checkpoint("Features", f"{x.shape[1]} columns")
    checkpoint("Feature list", ", ".join(x.columns))

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    checkpoint(
        "Train set",
        f"{X_train.shape[0]} samples ({X_train.shape[0] / len(x) * 100:.1f}%)",
    )
    checkpoint(
        "Test set", f"{X_test.shape[0]} samples ({X_test.shape[0] / len(x) * 100:.1f}%)"
    )
    checkpoint("Train class balance", f"{y_train.mean():.1%} malignant")
    checkpoint("Test class balance", f"{y_test.mean():.1%} malignant")

    # ── Scale features ──
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    train_means = X_train_scaled.mean().abs()
    train_stds = X_train_scaled.std()
    checkpoint(
        "Scaling verify (train)",
        f"mean range [{train_means.min():.4f}, {train_means.max():.4f}], std range [{train_stds.min():.4f}, {train_stds.max():.4f}]",
    )

    # ── Correlation check ──
    corr_matrix = X_train_scaled.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (col, upper_tri[col].idxmax(), upper_tri[col].max())
        for col in upper_tri.columns
        if any(upper_tri[col] > 0.9)
    ]
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    checkpoint("High correlation pairs (r > 0.9)", f"{len(high_corr_pairs)} found")
    for feat1, feat2, corr in high_corr_pairs:
        checkpoint("  Corr pair", f"{feat1} <-> {feat2} = {corr:.3f}")

    # ── Write summary file ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_text = f"Preprocessing Summary — {timestamp}\n{'=' * 50}\n"
    summary_text += "\n".join(line.replace("[QC] ", "") for line in summary_lines)
    summary_text += "\n"

    with open("preprocessing_summary.txt", "w") as f:
        f.write(summary_text)
    print(f"\n[QC] Summary written to preprocessing_summary.txt")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    run_preprocessing()
