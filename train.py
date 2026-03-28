# Training Script — XGBoost Breast Cancer Classifier
# ---------------------------------------------------
# Loads preprocessed data, trains an XGBoost model,
# evaluates performance, and saves artifacts to model/

import joblib
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from preprocess import run_preprocessing


def train():
    summary_lines = []

    def checkpoint(label, detail):
        """Print and log a QC checkpoint."""
        msg = f"[QC] {label}: {detail}"
        print(msg)
        summary_lines.append(msg)

    # ── Step 1: Load preprocessed data ──
    print("=" * 50)
    print("STEP 1 — Preprocessing")
    print("=" * 50)
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()

    # ── Step 2: Configure and train the model ──
    print("\n" + "=" * 50)
    print("STEP 2 — Training")
    print("=" * 50)

    # Calculate class weight to handle imbalance (more benign than malignant)
    # scale_pos_weight = number of negatives / number of positives
    n_benign = (y_train == 0).sum()
    n_malignant = (y_train == 1).sum()
    scale_weight = n_benign / n_malignant
    checkpoint("Class weight", f"{scale_weight:.3f} (benign={n_benign}, malignant={n_malignant})")

    # Create the XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,           # number of boosting rounds
        max_depth=5,                # max tree depth — keeps model from overfitting
        learning_rate=0.1,          # step size shrinkage
        scale_pos_weight=scale_weight,  # compensate for class imbalance
        random_state=42,
        eval_metric="logloss",      # binary cross-entropy loss
    )

    checkpoint("Model config", f"n_estimators=100, max_depth=5, lr=0.1")

    # Fit the model on the training data
    model.fit(X_train, y_train)
    checkpoint("Training", "Complete")

    # ── Step 3: Evaluate on the test set ──
    print("\n" + "=" * 50)
    print("STEP 3 — Evaluation")
    print("=" * 50)

    # Generate predictions and probability scores
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of malignant

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    checkpoint("Accuracy", f"{acc:.4f}")
    checkpoint("Precision", f"{prec:.4f}")
    checkpoint("Recall", f"{rec:.4f}")
    checkpoint("F1 Score", f"{f1:.4f}")
    checkpoint("ROC-AUC", f"{auc:.4f}")

    # Confusion matrix breakdown
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    checkpoint("Confusion matrix", f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Full classification report (precision, recall, f1 per class)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
    print(f"\n{report}")
    summary_lines.append(f"\nClassification Report:\n{report}")

    # ── Step 4: Save model and scaler ──
    print("=" * 50)
    print("STEP 4 — Saving Artifacts")
    print("=" * 50)

    # Save model in XGBoost's native JSON format (portable, human-readable)
    model.save_model("model/model.json")
    checkpoint("Model saved", "model/model.json")

    # Save the fitted scaler so the app can transform new inputs the same way
    joblib.dump(scaler, "model/scaler.pkl")
    checkpoint("Scaler saved", "model/scaler.pkl")

    # Save feature names so the app knows the expected input order
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, "model/feature_names.pkl")
    checkpoint("Feature names saved", f"model/feature_names.pkl ({len(feature_names)} features)")

    # ── Write training summary ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_text = f"Training Summary — {timestamp}\n{'=' * 50}\n"
    summary_text += "\n".join(line.replace("[QC] ", "") for line in summary_lines)
    summary_text += "\n"

    with open("training_summary.txt", "w") as f:
        f.write(summary_text)
    print(f"\n[QC] Summary written to training_summary.txt")


if __name__ == "__main__":
    train()
