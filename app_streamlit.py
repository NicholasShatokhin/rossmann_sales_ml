import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="🧮", layout="wide")

st.title("🧮 Rossmann Sales Forecast")
st.caption("Select a saved model bundle, enter features or upload a CSV, and get predictions.")

ART_DIR = Path("./models")

@st.cache_resource
def list_bundles(art_dir: Path):
    if not art_dir.exists():
        return []
    return [p for p in art_dir.iterdir() if p.is_dir() and (p / "model.joblib").exists() and (p / "preprocessor.joblib").exists() and (p / "meta.json").exists()]

@st.cache_resource
def load_bundle(bundle_dir: Path):
    model = joblib.load(bundle_dir / "model.joblib")
    preproc = joblib.load(bundle_dir / "preprocessor.joblib")
    with open(bundle_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, preproc, meta

def preprocess_input(df: pd.DataFrame, meta: dict, preproc):
    expected = list(meta["cat_features"]) + list(meta["num_features"])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[expected].copy()
    X_prep = preproc.transform(X)
    return X, X_prep

# Sidebar — select bundle
bundles = list_bundles(ART_DIR)
bundle_names = [b.name for b in bundles]
if not bundles:
    st.error("No bundles found in ./models. Please save your models first.")
    st.stop()

sel_name = st.sidebar.selectbox("Select model bundle", bundle_names, index=0)
bundle_dir = ART_DIR / sel_name
model, preproc, meta = load_bundle(bundle_dir)

st.sidebar.write("**Task type:**", meta.get("task_type", "?"))
st.sidebar.write("**Categorical features:**", ", ".join(meta.get("cat_features", [])))
st.sidebar.write("**Numeric features:**", ", ".join(meta.get("num_features", [])))

st.markdown("---")
tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab_single:
    st.subheader("Enter feature values")
    cols = st.columns(2)
    cat_vals = {}
    num_vals = {}

    cat_features = meta.get("cat_features", [])
    num_features = meta.get("num_features", [])

    with cols[0]:
        for c in cat_features:
            cat_vals[c] = st.text_input(f"{c}", value="")

    with cols[1]:
        for c in num_features:
            num_vals[c] = st.number_input(f"{c}", value=0.0, step=1.0, format="%.4f")

    if st.button("Get Prediction", type="primary"):
        try:
            row = {**cat_vals, **num_vals}
            df_in = pd.DataFrame([row])
            _, X_prep = preprocess_input(df_in, meta, preproc)
            y_hat = model.predict(X_prep)
            if meta.get("task_type") == "classification":
                st.success(f"Prediction (class): {int(y_hat[0])}")
            else:
                st.success(f"Prediction (sales): {float(y_hat[0]):,.2f}")
            st.dataframe(df_in.assign(prediction=y_hat))
        except Exception as e:
            st.error(f"Failed to predict: {e}")

with tab_batch:
    st.subheader("Upload a CSV with the expected columns")
    st.caption("Columns must match the training features: cat_features + num_features.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        try:
            df_csv = pd.read_csv(file)
            X_raw, X_prep = preprocess_input(df_csv, meta, preproc)
            y_hat = model.predict(X_prep)
            out = df_csv.copy()
            out["prediction"] = y_hat
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50))
            st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"),
                               file_name=f"{sel_name}_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

st.markdown("---")
st.caption("Tip: The app uses One-Hot Encoder with handle_unknown='ignore', so unseen categorical values are safely ignored.")