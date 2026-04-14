from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.business import top_share_flags
from src.config import MODELS_DIR
from src.data import basic_cleaning
from src.features import align_features_to_training, prepare_features_and_target


def parse_args():
    parser = argparse.ArgumentParser(description="Score new leads with the trained model.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output scored CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    model_bundle = joblib.load(MODELS_DIR / "lead_scoring_model.joblib")
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    recommended_contact_share = float(model_bundle["recommended_contact_share"])
    validation_threshold = float(model_bundle["validation_threshold"])

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    cleaned_df = basic_cleaning(raw_df)

    X_input, _ = prepare_features_and_target(cleaned_df)
    X_aligned, extra_columns = align_features_to_training(X_input, feature_columns)

    scores = model.predict_proba(X_aligned)[:, 1]

    scored_df = raw_df.copy()
    scored_df["lead_score"] = scores
    scored_df["lead_rank"] = scored_df["lead_score"].rank(method="first", ascending=False).astype(int)
    scored_df["lead_percentile"] = (scored_df["lead_rank"] / len(scored_df)).round(4)
    scored_df["recommend_contact_top_share"] = top_share_flags(scores, recommended_contact_share)
    scored_df["recommend_contact_threshold"] = (scores >= validation_threshold).astype(int)

    scored_df = scored_df.sort_values("lead_score", ascending=False)
    scored_df.to_csv(output_path, index=False)

    print(f"Scored {len(scored_df)} rows.")
    print(f"Saved output to: {output_path}")

    if extra_columns:
        print(f"Ignored extra input columns not used by the model: {extra_columns}")


if __name__ == "__main__":
    main()