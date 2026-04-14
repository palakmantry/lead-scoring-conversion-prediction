from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from src.business import build_policy_table, evaluate_contact_share, plot_gain_curve, top_share_flags
from src.config import FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, RANDOM_STATE, REPORTS_DIR
from src.data import ensure_directories, prepare_base_dataframe, split_chronological
from src.features import prepare_features_and_target
from src.modeling import classification_metrics, save_json


def plot_precision_recall(y_true, scores, output_path):
    precision, recall, _ = precision_recall_curve(y_true, scores)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_calibration(y_true, scores, output_path):
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")

    plt.figure(figsize=(8, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed conversion rate")
    plt.title("Calibration Curve (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_distribution(y_true, scores, output_path):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    plt.figure(figsize=(8, 5))
    plt.hist(scores[y_true == 0], bins=30, alpha=0.7, label="No conversion")
    plt.hist(scores[y_true == 1], bins=30, alpha=0.7, label="Conversion")
    plt.xlabel("Predicted conversion probability")
    plt.ylabel("Count")
    plt.title("Score Distribution by Actual Class (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(model, X_test, y_test, output_path, top_n=15):
    sample_n = min(5000, len(X_test))
    sample_idx = X_test.sample(n=sample_n, random_state=RANDOM_STATE).index

    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]

    result = permutation_importance(
        estimator=model,
        X=X_sample,
        y=y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance": result.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances (Test)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return importance_df


def main():
    ensure_directories()

    model_bundle = joblib.load(MODELS_DIR / "lead_scoring_model.joblib")
    model = model_bundle["model"]
    recommended_contact_share = float(model_bundle["recommended_contact_share"])
    validation_threshold = float(model_bundle["validation_threshold"])

    df = prepare_base_dataframe()
    _, _, test_df = split_chronological(df)

    X_test, y_test = prepare_features_and_target(test_df)
    test_scores = model.predict_proba(X_test)[:, 1]

    # Core metrics
    test_metrics = classification_metrics(y_test, test_scores)

    # Business policy evaluation
    test_policy_df = build_policy_table(y_test, test_scores)
    chosen_policy_test = evaluate_contact_share(
        y_true=y_test,
        scores=test_scores,
        share=recommended_contact_share,
    )

    # Threshold-based analysis using validation threshold
    threshold_flags = (test_scores >= validation_threshold).astype(int)
    threshold_contact_rate = float(threshold_flags.mean())
    threshold_contacted = int(threshold_flags.sum())
    threshold_conversions = int(y_test[threshold_flags == 1].sum()) if threshold_contacted > 0 else 0
    threshold_precision = (
        float(y_test[threshold_flags == 1].mean()) if threshold_contacted > 0 else 0.0
    )
    threshold_recall = (
        float(threshold_conversions / y_test.sum()) if y_test.sum() > 0 else 0.0
    )

    # Save scored test leads
    test_scored = test_df.copy()
    test_scored["lead_score"] = test_scores
    test_scored["lead_rank"] = test_scored["lead_score"].rank(method="first", ascending=False).astype(int)
    test_scored["recommend_contact_top_share"] = top_share_flags(test_scores, recommended_contact_share)
    test_scored["recommend_contact_threshold"] = (test_scores >= validation_threshold).astype(int)
    test_scored = test_scored.sort_values("lead_score", ascending=False)
    test_scored.to_csv(PREDICTIONS_DIR / "test_scored_leads.csv", index=False)

    # Save policy table
    test_policy_df.to_csv(REPORTS_DIR / "test_contact_policy.csv", index=False)

    # Save figures
    plot_precision_recall(
        y_true=y_test,
        scores=test_scores,
        output_path=FIGURES_DIR / "04_precision_recall_curve_test.png",
    )
    plot_calibration(
        y_true=y_test,
        scores=test_scores,
        output_path=FIGURES_DIR / "05_calibration_curve_test.png",
    )
    plot_score_distribution(
        y_true=y_test,
        scores=test_scores,
        output_path=FIGURES_DIR / "06_score_distribution_test.png",
    )
    importance_df = plot_feature_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        output_path=FIGURES_DIR / "07_feature_importance_test.png",
    )
    importance_df.to_csv(REPORTS_DIR / "test_feature_importance.csv", index=False)

    plot_gain_curve(
        y_true=y_test,
        scores=test_scores,
        output_path=FIGURES_DIR / "08_gain_curve_test.png",
    )

    evaluation_summary = {
        "test_metrics": test_metrics,
        "recommended_contact_share_from_validation": recommended_contact_share,
        "chosen_policy_test_results": chosen_policy_test,
        "threshold_based_results_using_validation_threshold": {
            "validation_threshold": validation_threshold,
            "contact_rate": threshold_contact_rate,
            "contacted_leads": threshold_contacted,
            "captured_conversions": threshold_conversions,
            "precision": threshold_precision,
            "recall": threshold_recall,
        },
    }

    save_json(evaluation_summary, REPORTS_DIR / "test_metrics.json")

    print("Evaluation complete.")
    print(f"Saved metrics to: {REPORTS_DIR / 'test_metrics.json'}")
    print(f"Saved scored test leads to: {PREDICTIONS_DIR / 'test_scored_leads.csv'}")
    print(f"Saved charts to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()