# app_streamlit.py — updated
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import zipfile
import tempfile

st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="🧮", layout="wide")

st.title("🧮 Rossmann Sales Forecast")
st.caption("Select a saved model bundle, or load one via Upload / Google Drive, then enter features or upload a CSV to get predictions.")

ART_DIR = Path("./models")

# ---------- Local bundles ----------
@st.cache_resource
def list_bundles(art_dir: Path):
    if not art_dir.exists():
        return []
    return [
        p for p in art_dir.iterdir()
        if p.is_dir()
        and (p / "model.joblib").exists()
        and (p / "preprocessor.joblib").exists()
        and (p / "meta.json").exists()
    ]

@st.cache_resource
def load_bundle_from_dir(bundle_dir: Path):
    model = joblib.load(bundle_dir / "model.joblib")
    preproc = joblib.load(bundle_dir / "preprocessor.joblib")
    with open(bundle_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, preproc, meta

# ---------- Upload / Google Drive ----------
def _normalize_bundle(obj: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Accepts a dict-like joblib bundle and returns (model, preproc, meta).
    Expected keys (aliases allowed):
      model: 'model'
      preprocessor: 'preprocessor' or 'preproc'
      meta: 'meta' (dict with 'cat_features' and 'num_features')
    """
    if not isinstance(obj, dict):
        raise ValueError("Uploaded bundle must be a dict with keys: 'model', 'preprocessor', 'meta'.")
    model = obj.get("model")
    preproc = obj.get("preprocessor", obj.get("preproc"))
    meta = obj.get("meta")
    if model is None or preproc is None or meta is None:
        raise ValueError("Bundle dict missing required keys: 'model', 'preprocessor', and 'meta'.")
    # Basic validation
    if "cat_features" not in meta or "num_features" not in meta:
        raise ValueError("meta must contain 'cat_features' and 'num_features'.")
    return model, preproc, meta

def load_bundle_from_bytes_or_zip(file_bytes: bytes, filename: str="") -> Tuple[Any, Any, Dict[str, Any]]:
    """Takes one .joblib-dictionary, or .zip with three files."""
    # if zip — read three files
    if filename.lower().endswith(".zip") or zipfile.is_zipfile(io.BytesIO(file_bytes)):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = {n.lower(): n for n in zf.namelist()}
            need = ["model.joblib", "preprocessor.joblib", "meta.json"]
            missing = [n for n in need if n not in names]
            if missing:
                raise ValueError(f"ZIP missing required files: {missing}")
            with zf.open(names["model.joblib"]) as f:
                model = joblib.load(f)
            with zf.open(names["preprocessor.joblib"]) as f:
                preproc = joblib.load(f)
            with zf.open(names["meta.json"]) as f:
                meta = json.load(f)
            return model, preproc, meta
    # else try as .joblib-dictionary
    obj = joblib.load(io.BytesIO(file_bytes))
    return _normalize_bundle(obj)

def load_bundle_from_uploads(files: List[Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    """Takes one .joblib (dict), or set of 3 files, or one .zip."""
    # Case 1: one file
    if len(files) == 1:
        f = files[0]
        b = f.read()
        return load_bundle_from_bytes_or_zip(b, f.name)
    # Case 2: few files — seek for three needed
    by_name = {f.name.lower(): f for f in files}
    need = ["model.joblib", "preprocessor.joblib", "meta.json"]
    if all(n in by_name for n in need):
        model = joblib.load(by_name["model.joblib"])
        preproc = joblib.load(by_name["preprocessor.joblib"])
        meta = json.load(by_name["meta.json"])
        return model, preproc, meta
    raise ValueError("Provide either one .joblib dict, one .zip, or the three files: model.joblib, preprocessor.joblib, meta.json.")

def parse_gdrive_id(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    if "drive.google.com" in text:
        import re
        m = re.search(r"/d/([A-Za-z0-9_-]+)", text)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", text)
        if m:
            return m.group(1)
        return None
    # Treat as raw ID
    return text

@st.cache_data(show_spinner=False)
def download_from_gdrive(file_id: str) -> Tuple[bytes, str]:
    import requests
    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(URL, params=params, stream=True, timeout=None)
    # Confirm token for large files
    def _get_confirm_token(resp):
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None
    token = _get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(URL, params=params, stream=True, timeout=60)
    response.raise_for_status()
    # Read in stream
    content = io.BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            content.write(chunk)
    # Get filename from headers (if exists)
    cd = response.headers.get("Content-Disposition", "")
    fname = ""
    if "filename=" in cd:
        fname = cd.split("filename=")[-1].strip('"; ')
    return content.getvalue(), fname or f"{file_id}.bin"

# ---------- Preprocess ----------
def preprocess_input(df: pd.DataFrame, meta: dict, preproc):
    expected = list(meta["cat_features"]) + list(meta["num_features"])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[expected].copy()
    X_prep = preproc.transform(X)
    return X, X_prep

# ---------- Sidebar: Model source ----------
with st.sidebar:
    st.header("Load Model")
    src_tab_local, src_tab_upload, src_tab_gdrive = st.tabs(["Local bundles", "Upload", "Google Drive"])

    bundle_loaded = False
    model = preproc = meta = None

    # Local bundles (original behavior)
    with src_tab_local:
        bundles = list_bundles(ART_DIR)
        if bundles:
            bundle_names = [b.name for b in bundles]
            sel_name = st.selectbox("Select local model bundle", bundle_names, index=0)
            if st.button("Load selected bundle", use_container_width=True):
                model, preproc, meta = load_bundle_from_dir(ART_DIR / sel_name)
                bundle_loaded = True
        else:
            st.info("No bundles found in ./models. You can upload a .joblib bundle or use Google Drive.")

    # Upload
    with src_tab_upload:
        st.caption("Upload: (a) one .joblib-dictionary OR (b) three files: model.joblib, preprocessor.joblib, meta.json OR (c) one .zip with that three files.")
        ups = st.file_uploader("Upload files", type=["joblib","pkl","json","zip"], accept_multiple_files=True)
        if ups and st.button("Load uploaded file(s)", use_container_width=True):
            try:
                model, preproc, meta = load_bundle_from_uploads(ups)
                bundle_loaded = True
                st.success("Bundle loaded from uploaded file(s).")
            except Exception as e:
                st.error(f"Failed to load uploaded bundle: {e}")

    # Google Drive
    with src_tab_gdrive:
        gd_text = st.text_input("Paste Google Drive link or File ID (single file: .joblib or .zip)", value="")
        if st.button("Download & load from Drive", use_container_width=True):
            try:
                fid = parse_gdrive_id(gd_text)
                if not fid:
                    raise ValueError("Could not parse Google Drive file ID.")
                content, filename = download_from_gdrive(fid)
                model, preproc, meta = load_bundle_from_bytes_or_zip(content, filename)
                bundle_loaded = True
                st.success("Bundle loaded from Google Drive.")
            except Exception as e:
                st.error(f"Google Drive load failed: {e}")

if not bundle_loaded:
    st.stop()

# Sidebar model info
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
                               file_name="rossmann_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

st.markdown("---")
st.caption("Tip: The app uses One-Hot Encoder with handle_unknown='ignore', so unseen categorical values are safely ignored.")
