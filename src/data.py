from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_URL,
    FIGURES_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    RAW_DIR,
    REPORTS_DIR,
    TRAIN_FRACTION,
    VALID_FRACTION,
)


DIRECTORIES = [
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
]


def ensure_directories() -> None:
    """Create all required project directories."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for easier code usage."""
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(".", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning used by train / evaluate / predict."""
    df = standardize_columns(df)

    object_cols = df.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    if "y" in df.columns:
        mapped = df["y"].astype(str).str.lower().map({"yes": 1, "no": 0})
        if mapped.notna().all():
            df["y"] = mapped.astype(int)
        else:
            df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)

    return df


def download_dataset(force: bool = False) -> Path:
    """
    Download the UCI Bank Marketing archive, handle nested zip files,
    extract bank-additional-full.csv, normalize it, and save a clean CSV
    to data/raw/bank_marketing_full.csv.
    """
    ensure_directories()

    if RAW_DATA_PATH.exists() and not force:
        print(f"Dataset already exists at: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "bank_marketing.zip"

        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)

        print("Extracting outer archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        csv_candidates = list(tmpdir_path.rglob("bank-additional-full.csv"))

        if not csv_candidates:
            nested_zip_files = list(tmpdir_path.rglob("*.zip"))

            for nested_zip in nested_zip_files:
                nested_extract_dir = nested_zip.parent / nested_zip.stem
                nested_extract_dir.mkdir(parents=True, exist_ok=True)

                try:
                    with zipfile.ZipFile(nested_zip, "r") as zf:
                        zf.extractall(nested_extract_dir)
                except zipfile.BadZipFile:
                    continue

            csv_candidates = list(tmpdir_path.rglob("bank-additional-full.csv"))

        if not csv_candidates:
            raise FileNotFoundError(
                "Could not find bank-additional-full.csv even after extracting nested zip files."
            )

        df = pd.read_csv(csv_candidates[0], sep=";")
        df = basic_cleaning(df)
        df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Saved cleaned raw dataset to: {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    """Load the normalized raw dataset saved by download_dataset()."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Run: python -m src.download_data"
        )
    return pd.read_csv(RAW_DATA_PATH)


def prepare_base_dataframe() -> pd.DataFrame:
    """Load + clean base dataframe."""
    return basic_cleaning(load_raw_data())


def split_chronological(df: pd.DataFrame):
    """
    Split in original row order.
    The chosen dataset is already ordered by time, so this simulates
    training on historical data and testing on later data.
    """
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_FRACTION)
    valid_end = int(n_rows * (TRAIN_FRACTION + VALID_FRACTION))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df