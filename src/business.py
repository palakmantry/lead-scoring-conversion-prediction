from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CONTACT_COST, CONTACT_SHARE_GRID, CONVERSION_VALUE


def top_share_flags(scores, share: float) -> np.ndarray:
    """
    Mark the top X% highest-scoring rows as 1, everything else as 0.
    """
    scores = np.asarray(scores)
    n_contact = max(1, int(np.ceil(len(scores) * share)))

    order = np.argsort(-scores)
    flags = np.zeros(len(scores), dtype=int)
    flags[order[:n_contact]] = 1
    return flags


def threshold_for_share(scores, share: float) -> float:
    """
    Return the score threshold that corresponds to the chosen contact share.
    """
    scores = np.asarray(scores)
    n_contact = max(1, int(np.ceil(len(scores) * share)))
    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[n_contact - 1])


def evaluate_contact_share(
    y_true,
    scores,
    share: float,
    contact_cost: int = CONTACT_COST,
    conversion_value: int = CONVERSION_VALUE,
) -> dict:
    """
    Evaluate business results if we contact only the top X% of leads.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    flags = top_share_flags(scores, share)
    contacted = int(flags.sum())
    conversions = int(y_true[flags == 1].sum())

    precision = float(conversions / contacted) if contacted > 0 else 0.0
    recall = float(conversions / y_true.sum()) if y_true.sum() > 0 else 0.0

    revenue = int(conversions * conversion_value)
    cost = int(contacted * contact_cost)
    profit = int(revenue - cost)
    roi = float(profit / cost) if cost > 0 else 0.0

    return {
        "contact_share": float(share),
        "threshold": float(threshold_for_share(scores, share)),
        "contacted_leads": contacted,
        "captured_conversions": conversions,
        "precision": precision,
        "recall": recall,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "roi": roi,
    }


def build_policy_table(
    y_true,
    scores,
    shares: list[float] | None = None,
    contact_cost: int = CONTACT_COST,
    conversion_value: int = CONVERSION_VALUE,
) -> pd.DataFrame:
    """
    Build a table of business results across multiple contact-share policies.
    """
    if shares is None:
        shares = CONTACT_SHARE_GRID

    rows = [
        evaluate_contact_share(
            y_true=y_true,
            scores=scores,
            share=share,
            contact_cost=contact_cost,
            conversion_value=conversion_value,
        )
        for share in shares
    ]

    return pd.DataFrame(rows).sort_values("contact_share").reset_index(drop=True)


def select_best_policy(policy_df: pd.DataFrame) -> dict:
    """
    Pick the best business policy.
    Current rule: choose the highest-profit option.
    """
    best_row = policy_df.sort_values(["profit", "precision"], ascending=[False, False]).iloc[0]
    return best_row.to_dict()


def plot_gain_curve(y_true, scores, output_path) -> None:
    """
    Plot cumulative recall captured as we contact more leads.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    cumulative_positives = np.cumsum(y_sorted)
    total_positives = y_sorted.sum()

    population_share = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    captured_share = (
        cumulative_positives / total_positives if total_positives > 0 else np.zeros_like(cumulative_positives)
    )

    plt.figure(figsize=(8, 5))
    plt.plot(population_share, captured_share, label="Model gain")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("Share of leads contacted")
    plt.ylabel("Share of conversions captured")
    plt.title("Cumulative Gain Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()