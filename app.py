# Streamlit Dashboard for Breast Cancer Prediction
# -------------------------------------------------
# Loads the trained XGBoost model and scaler, lets the user
# input cell measurements, and displays a prediction with
# confidence and feature importance.

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# ── Page config ──
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🔬",
    layout="wide",
)


# ── Load model artifacts (cached so they only load once) ──
@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model("model/model.json")
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

# ── Feature metadata ──
# Readable labels, min/max/default values from the training dataset
# Features are grouped into three categories: mean, standard error, worst
FEATURE_INFO = {
    "radius_mean": {
        "label": "Radius",
        "min": 6.0,
        "max": 30.0,
        "default": 14.13,
        "step": 0.01,
        "format": "%.2f",
    },
    "smoothness_mean": {
        "label": "Smoothness",
        "min": 0.05,
        "max": 0.17,
        "default": 0.0964,
        "step": 0.0001,
        "format": "%.4f",
    },
    "compactness_mean": {
        "label": "Compactness",
        "min": 0.01,
        "max": 0.35,
        "default": 0.1043,
        "step": 0.001,
        "format": "%.4f",
    },
    "concavity_mean": {
        "label": "Concavity",
        "min": 0.0,
        "max": 0.45,
        "default": 0.0888,
        "step": 0.001,
        "format": "%.4f",
    },
    "symmetry_mean": {
        "label": "Symmetry",
        "min": 0.10,
        "max": 0.31,
        "default": 0.1812,
        "step": 0.001,
        "format": "%.4f",
    },
    "fractal_dimension_mean": {
        "label": "Fractal Dimension",
        "min": 0.04,
        "max": 0.10,
        "default": 0.0628,
        "step": 0.0001,
        "format": "%.4f",
    },
    "radius_se": {
        "label": "Radius",
        "min": 0.1,
        "max": 3.0,
        "default": 0.4052,
        "step": 0.001,
        "format": "%.4f",
    },
    "texture_se": {
        "label": "Texture",
        "min": 0.3,
        "max": 5.0,
        "default": 1.2169,
        "step": 0.001,
        "format": "%.4f",
    },
    "smoothness_se": {
        "label": "Smoothness",
        "min": 0.001,
        "max": 0.032,
        "default": 0.007,
        "step": 0.0001,
        "format": "%.4f",
    },
    "compactness_se": {
        "label": "Compactness",
        "min": 0.002,
        "max": 0.14,
        "default": 0.0255,
        "step": 0.001,
        "format": "%.4f",
    },
    "concavity_se": {
        "label": "Concavity",
        "min": 0.0,
        "max": 0.40,
        "default": 0.0319,
        "step": 0.001,
        "format": "%.4f",
    },
    "concave points_se": {
        "label": "Concave Points",
        "min": 0.0,
        "max": 0.054,
        "default": 0.0118,
        "step": 0.0001,
        "format": "%.4f",
    },
    "symmetry_se": {
        "label": "Symmetry",
        "min": 0.007,
        "max": 0.08,
        "default": 0.0205,
        "step": 0.0001,
        "format": "%.4f",
    },
    "fractal_dimension_se": {
        "label": "Fractal Dimension",
        "min": 0.0008,
        "max": 0.03,
        "default": 0.0038,
        "step": 0.0001,
        "format": "%.4f",
    },
    "radius_worst": {
        "label": "Radius",
        "min": 7.0,
        "max": 37.0,
        "default": 16.27,
        "step": 0.01,
        "format": "%.2f",
    },
    "texture_worst": {
        "label": "Texture",
        "min": 12.0,
        "max": 50.0,
        "default": 25.68,
        "step": 0.01,
        "format": "%.2f",
    },
    "smoothness_worst": {
        "label": "Smoothness",
        "min": 0.07,
        "max": 0.23,
        "default": 0.1324,
        "step": 0.0001,
        "format": "%.4f",
    },
    "compactness_worst": {
        "label": "Compactness",
        "min": 0.02,
        "max": 1.1,
        "default": 0.2543,
        "step": 0.001,
        "format": "%.4f",
    },
    "concavity_worst": {
        "label": "Concavity",
        "min": 0.0,
        "max": 1.3,
        "default": 0.2722,
        "step": 0.001,
        "format": "%.4f",
    },
    "symmetry_worst": {
        "label": "Symmetry",
        "min": 0.15,
        "max": 0.67,
        "default": 0.2901,
        "step": 0.001,
        "format": "%.4f",
    },
    "fractal_dimension_worst": {
        "label": "Fractal Dimension",
        "min": 0.05,
        "max": 0.21,
        "default": 0.0839,
        "step": 0.0001,
        "format": "%.4f",
    },
}

# Split features into their three groups
MEAN_FEATURES = [f for f in feature_names if f.endswith("_mean")]
SE_FEATURES = [f for f in feature_names if f.endswith("_se")]
WORST_FEATURES = [f for f in feature_names if f.endswith("_worst")]


# ── Header ──
st.title("Breast Cancer Prediction")
st.markdown(
    "Enter cell nucleus measurements from a fine needle aspirate (FNA) to predict "
    "whether a breast mass is **benign** or **malignant**."
)

# ── Sidebar: input form ──
st.sidebar.header("Cell Measurements")
st.sidebar.caption("Adjust the sliders to match the biopsy measurements.")

input_values = {}


def render_feature_group(features, group_label):
    """Render a group of feature sliders under a subheader."""
    st.sidebar.subheader(group_label)
    for feat in features:
        info = FEATURE_INFO[feat]
        input_values[feat] = st.sidebar.number_input(
            label=info["label"],
            min_value=info["min"],
            max_value=info["max"],
            value=info["default"],
            step=info["step"],
            format=info["format"],
            key=feat,
        )


render_feature_group(MEAN_FEATURES, "Mean Values")
render_feature_group(SE_FEATURES, "Standard Error")
render_feature_group(WORST_FEATURES, "Worst (Largest) Values")

# ── Build input array in the correct feature order ──
input_array = np.array([[input_values[f] for f in feature_names]])

# Scale using the same scaler from training
input_scaled = scaler.transform(input_array)

# ── Prediction ──
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]
confidence = probabilities[prediction]

# ── Results layout ──
col_result, col_probs = st.columns([1, 1])

with col_result:
    st.subheader("Prediction")

    if prediction == 1:
        st.error(f"**Malignant** — {confidence:.1%} confidence")
    else:
        st.success(f"**Benign** — {confidence:.1%} confidence")

    st.caption(
        "This model achieved 96.5% accuracy and 0.99 ROC-AUC on the test set. "
        "It is not a substitute for professional medical diagnosis."
    )

with col_probs:
    st.subheader("Class Probabilities")

    # Horizontal bar showing benign vs malignant probability
    prob_df = pd.DataFrame(
        {
            "Class": ["Benign", "Malignant"],
            "Probability": [probabilities[0], probabilities[1]],
        }
    )
    st.bar_chart(prob_df.set_index("Class"), horizontal=True, height=150)

# ── Feature Importance ──
st.subheader("Feature Importance")
st.caption("Which measurements influence the model's prediction the most.")

# Get importance scores from the trained model
importance = model.feature_importances_
importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance,
    }
).sort_values("Importance", ascending=True)

# Show the top 15 most important features as a horizontal bar chart
top_n = importance_df.tail(15).set_index("Feature")
st.bar_chart(top_n, horizontal=True, height=450)

# ── Input summary (collapsible) ──
with st.expander("View raw input values"):
    input_df = pd.DataFrame([input_values])
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
