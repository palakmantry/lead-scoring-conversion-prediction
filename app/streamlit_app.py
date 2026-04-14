from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.business import top_share_flags
from src.data import basic_cleaning
from src.features import align_features_to_training, prepare_features_and_target


MODEL_PATH = ROOT / "models" / "lead_scoring_model.joblib"

st.set_page_config(page_title="Lead Scoring Demo", layout="wide")
st.title("Lead Scoring Demo")
st.write("Upload a CSV of leads and get ranked conversion scores.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run training first: python -m src.train")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
recommended_contact_share = float(bundle["recommended_contact_share"])
validation_threshold = float(bundle["validation_threshold"])

st.sidebar.header("Model policy")
st.sidebar.write(f"Recommended contact share: {recommended_contact_share:.0%}")
st.sidebar.write(f"Reference probability threshold: {validation_threshold:.4f}")

uploaded_file = st.file_uploader("Upload lead CSV", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    cleaned_df = basic_cleaning(raw_df)

    X_input, _ = prepare_features_and_target(cleaned_df)
    X_aligned, extra_columns = align_features_to_training(X_input, feature_columns)

    scores = model.predict_proba(X_aligned)[:, 1]

    scored_df = raw_df.copy()
    scored_df["lead_score"] = scores
    scored_df["lead_rank"] = scored_df["lead_score"].rank(method="first", ascending=False).astype(int)
    scored_df["recommend_contact_top_share"] = top_share_flags(scores, recommended_contact_share)
    scored_df["recommend_contact_threshold"] = (scores >= validation_threshold).astype(int)
    scored_df = scored_df.sort_values("lead_score", ascending=False)

    st.subheader("Scored leads")
    st.dataframe(scored_df.head(50), use_container_width=True)

    if extra_columns:
        st.info(f"Extra columns ignored by the model: {extra_columns}")

    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored leads CSV",
        data=csv_bytes,
        file_name="scored_leads.csv",
        mime="text/csv",
    )
else:
    st.write("Expected input: a CSV with columns similar to the training dataset.")