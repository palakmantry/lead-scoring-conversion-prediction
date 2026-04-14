from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.data import ensure_directories, prepare_base_dataframe, split_chronological
from src.features import engineer_features
from src.modeling import save_json
from src.config import FIGURES_DIR, REPORTS_DIR


MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["y"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["No Conversion", "Conversion"], counts.values)
    plt.title("Target Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=200)
    plt.close()


def plot_monthly_conversion_rate(df: pd.DataFrame) -> None:
    monthly = df.groupby("month")["y"].mean().reindex(MONTH_ORDER)

    plt.figure(figsize=(8, 4))
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.title("Monthly Conversion Rate")
    plt.xlabel("Month")
    plt.ylabel("Conversion Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_monthly_conversion_rate.png", dpi=200)
    plt.close()


def plot_contact_type_conversion(df: pd.DataFrame) -> None:
    contact_conv = df.groupby("contact")["y"].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 4))
    plt.bar(contact_conv.index.astype(str), contact_conv.values)
    plt.title("Conversion Rate by Contact Type")
    plt.xlabel("Contact Type")
    plt.ylabel("Conversion Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_contact_type_conversion.png", dpi=200)
    plt.close()


def save_unknown_value_share(df: pd.DataFrame) -> None:
    rows = []

    for col in df.select_dtypes(include="object").columns:
        unknown_share = (df[col].astype(str).str.lower() == "unknown").mean()
        rows.append({"column": col, "unknown_share": round(float(unknown_share), 4)})

    unknown_df = pd.DataFrame(rows).sort_values("unknown_share", ascending=False)
    unknown_df.to_csv(REPORTS_DIR / "unknown_value_share.csv", index=False)


def save_eda_summary(df: pd.DataFrame) -> None:
    train_df, valid_df, test_df = split_chronological(df)

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "positive_rate": float(df["y"].mean()),
        "positive_cases": int(df["y"].sum()),
        "negative_cases": int((1 - df["y"]).sum()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "notes": [
            "Target is conversion / subscription yes-no",
            "Chronological split is used instead of random split",
            "The leakage column 'duration' will be dropped during modeling",
        ],
    }
    save_json(summary, REPORTS_DIR / "eda_summary.json")


def main():
    ensure_directories()

    df = prepare_base_dataframe()
    _ = engineer_features(df)

    plot_target_distribution(df)
    plot_monthly_conversion_rate(df)
    plot_contact_type_conversion(df)
    save_unknown_value_share(df)
    save_eda_summary(df)

    print("EDA complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"EDA summary saved to: {REPORTS_DIR / 'eda_summary.json'}")


if __name__ == "__main__":
    main()