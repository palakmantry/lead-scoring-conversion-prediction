from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LEAKAGE_COLUMNS, TARGET_COL


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a few business-friendly derived features.
    Keep these lightweight and explainable.
    """
    df = df.copy()

    if "pdays" in df.columns:
        df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)
        df["pdays_clean"] = df["pdays"].replace(999, np.nan)

    if "previous" in df.columns:
        df["had_previous_contacts"] = (df["previous"] > 0).astype(int)

    if "campaign" in df.columns:
        df["campaign_bucket"] = pd.cut(
            df["campaign"],
            bins=[-np.inf, 1, 2, 4, 7, np.inf],
            labels=["1", "2", "3-4", "5-7", "8+"],
        ).astype(str)

    if "contact" in df.columns:
        df["is_cellular"] = (df["contact"].astype(str).str.lower() == "cellular").astype(int)

    if "month" in df.columns:
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        df["month_num"] = df["month"].astype(str).str.lower().map(month_map)

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Return X, y after feature engineering and leakage removal.
    """
    df = engineer_features(df)

    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int).copy()
        X = df.drop(columns=[TARGET_COL], errors="ignore")
    else:
        X = df.copy()

    X = X.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    return X, y


def align_features_to_training(X: pd.DataFrame, feature_columns: list[str]):
    """
    Align scoring data to the exact feature schema used in training.
    Missing columns are added as NaN. Extra columns are ignored.
    """
    X = X.copy()
    extra_columns = [col for col in X.columns if col not in feature_columns]

    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_columns]
    return X, extra_columns