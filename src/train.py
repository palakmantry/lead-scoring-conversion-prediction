from __future__ import annotations

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.business import build_policy_table, select_best_policy, threshold_for_share
from src.config import MODELS_DIR, REPORTS_DIR
from src.data import ensure_directories, prepare_base_dataframe, split_chronological
from src.features import prepare_features_and_target
from src.modeling import (
    build_baseline_pipeline,
    build_random_forest_pipeline,
    classification_metrics,
    save_json,
    tune_random_forest,
)


def strip_model_prefix(best_params: dict) -> dict:
    """Convert RandomizedSearchCV params into RandomForest params."""
    return {key.replace("model__", ""): value for key, value in best_params.items()}


def main():
    ensure_directories()

    # 1) Load and split data
    df = prepare_base_dataframe()
    train_df, valid_df, test_df = split_chronological(df)

    X_train, y_train = prepare_features_and_target(train_df)
    X_valid, y_valid = prepare_features_and_target(valid_df)

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_valid, y_train_valid = prepare_features_and_target(train_valid_df)

    # 2) Baseline model
    baseline_model = build_baseline_pipeline(X_train)
    baseline_model.fit(X_train, y_train)
    baseline_valid_scores = baseline_model.predict_proba(X_valid)[:, 1]
    baseline_metrics = classification_metrics(y_valid, baseline_valid_scores)

    # 3) Improved model with tuning
    rf_search = tune_random_forest(X_train, y_train)
    tuned_rf_model = rf_search.best_estimator_
    tuned_rf_valid_scores = tuned_rf_model.predict_proba(X_valid)[:, 1]
    tuned_rf_metrics = classification_metrics(y_valid, tuned_rf_valid_scores)

    # 4) Probability-calibrated model
    calibrated_model = CalibratedClassifierCV(
        estimator=tuned_rf_model,
        method="sigmoid",
        cv=3,
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_valid_scores = calibrated_model.predict_proba(X_valid)[:, 1]
    calibrated_metrics = classification_metrics(y_valid, calibrated_valid_scores)

    # 5) Business policy selection on validation set
    policy_df = build_policy_table(y_valid, calibrated_valid_scores)
    best_policy = select_best_policy(policy_df)
    recommended_contact_share = float(best_policy["contact_share"])
    recommended_threshold = float(
        threshold_for_share(calibrated_valid_scores, recommended_contact_share)
    )

    # 6) Save validation comparison tables
    model_comparison_df = pd.DataFrame(
        [
            {"model": "baseline_logistic_regression", **baseline_metrics},
            {"model": "tuned_random_forest", **tuned_rf_metrics},
            {"model": "calibrated_random_forest", **calibrated_metrics},
        ]
    )
    model_comparison_df.to_csv(REPORTS_DIR / "validation_model_comparison.csv", index=False)
    policy_df.to_csv(REPORTS_DIR / "validation_contact_policy.csv", index=False)

    # 7) Refit final model on train + validation
    best_rf_params = strip_model_prefix(rf_search.best_params_)
    final_rf_model = build_random_forest_pipeline(X_train_valid, params=best_rf_params)

    final_calibrated_model = CalibratedClassifierCV(
        estimator=final_rf_model,
        method="sigmoid",
        cv=3,
    )
    final_calibrated_model.fit(X_train_valid, y_train_valid)

    model_bundle = {
        "model": final_calibrated_model,
        "feature_columns": list(X_train_valid.columns),
        "recommended_contact_share": recommended_contact_share,
        "validation_threshold": recommended_threshold,
        "best_random_forest_params": best_rf_params,
        "validation_metrics": {
            "baseline_logistic_regression": baseline_metrics,
            "tuned_random_forest": tuned_rf_metrics,
            "calibrated_random_forest": calibrated_metrics,
        },
        "validation_best_policy": best_policy,
        "split_sizes": {
            "train_rows": len(train_df),
            "validation_rows": len(valid_df),
            "test_rows": len(test_df),
        },
    }

    model_path = MODELS_DIR / "lead_scoring_model.joblib"
    joblib.dump(model_bundle, model_path)

    training_summary = {
        "saved_model_path": str(model_path),
        "best_random_forest_params": best_rf_params,
        "recommended_contact_share": recommended_contact_share,
        "validation_threshold": recommended_threshold,
        "validation_metrics": model_bundle["validation_metrics"],
        "validation_best_policy": best_policy,
    }
    save_json(training_summary, REPORTS_DIR / "training_summary.json")

    print("Training complete.")
    print(f"Saved final model to: {model_path}")
    print(f"Saved validation comparison to: {REPORTS_DIR / 'validation_model_comparison.csv'}")
    print(f"Saved contact policy to: {REPORTS_DIR / 'validation_contact_policy.csv'}")


if __name__ == "__main__":
    main()