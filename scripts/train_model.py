"""Train, validate, and export a reproducible CatBoost ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from climate_health.features import TARGET, build_features, feature_groups, load_competition_data, split_feature_types
from climate_health.postprocess import build_submission


SEEDS = (17,)


def official_score(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    label = (probability >= 0.5).astype(int)
    f1 = f1_score(y_true, label)
    auc = roc_auc_score(y_true, probability)
    return {"f1_default_0_5": float(f1), "roc_auc": float(auc), "weighted_score": float(0.6 * f1 + 0.4 * auc)}


def make_model(seed: int, iterations: int = 350, params: dict | None = None) -> CatBoostClassifier:
    params = params or {}
    return CatBoostClassifier(
        iterations=int(params.get("iterations", iterations)),
        depth=int(params.get("depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.035)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 6.0)),
        loss_function="Logloss",
        eval_metric="AUC",
        random_strength=float(params.get("random_strength", 0.5)),
        bagging_temperature=0.2,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def run_training(data_dir: str | Path = ".", report_dir: str | Path = "reports", submission_dir: str | Path = "submissions", model_dir: str | Path = "models", use_selected: bool = True, run_name: str = "catboost_ensemble") -> dict:
    train, test, climate = load_competition_data(data_dir)
    y = train[TARGET].to_numpy(dtype=int)
    raw_features = train.drop(columns=[TARGET])
    combined_raw = pd.concat([raw_features, test], ignore_index=True)
    X = build_features(raw_features, climate, combined_for_counts=combined_raw)
    X_test = build_features(test, climate, combined_for_counts=combined_raw)
    best_params_path = Path(model_dir) / "best_params.json"
    params = {}
    if best_params_path.exists():
        with best_params_path.open("r", encoding="utf-8") as handle:
            params = json.load(handle)
    selected_path = Path(model_dir) / "selected_features.json"
    selected_features = X.columns.tolist()
    if use_selected and selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            selected_features = json.load(handle)["features"]
        selected_features = [col for col in selected_features if col in X.columns]
        X = X[selected_features]
        X_test = X_test[selected_features]
    numeric, categorical = split_feature_types(X)
    cat_indices = [X.columns.get_loc(col) for col in categorical]

    report_path = Path(report_dir)
    table_path = report_path / "tables"
    submission_path = Path(submission_dir)
    model_path = Path(model_dir)
    table_path.mkdir(parents=True, exist_ok=True)
    submission_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_by_seed = []
    test_by_seed = []
    fold_rows = []
    importance_rows = []

    for seed in SEEDS:
        oof = np.zeros(len(train), dtype=float)
        test_pred = np.zeros(len(test), dtype=float)
        for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            model = make_model(seed, params=params)
            model.fit(X.iloc[fit_idx], y[fit_idx], cat_features=cat_indices)
            valid_pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            fold_test = model.predict_proba(X_test)[:, 1]
            oof[valid_idx] = valid_pred
            test_pred += fold_test / cv.n_splits
            metrics = official_score(y[valid_idx], valid_pred)
            metrics.update({"seed": seed, "fold": fold, "n_train": len(fit_idx), "n_valid": len(valid_idx)})
            fold_rows.append(metrics)
            importance_rows.append(pd.DataFrame({"feature": X.columns, "importance": model.get_feature_importance(), "seed": seed, "fold": fold}))
        oof_by_seed.append(oof)
        test_by_seed.append(test_pred)

    oof = np.mean(oof_by_seed, axis=0)
    test_probability = np.mean(test_by_seed, axis=0)
    overall = official_score(y, oof)
    overall.update({"n_train": int(len(train)), "n_test": int(len(test)), "n_features": int(X.shape[1]), "n_categorical": int(len(categorical)), "seeds": list(SEEDS)})

    pd.DataFrame(fold_rows).to_csv(table_path / "cv_fold_results.csv", index=False)
    importance = pd.concat(importance_rows, ignore_index=True).groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    importance.to_csv(table_path / "feature_importance.csv", index=False)
    group_map = feature_groups(X.columns)
    group_lookup = {feature: group for group, features in group_map.items() for feature in features}
    importance["group"] = importance["feature"].map(group_lookup).fillna("other")
    importance.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False).to_csv(table_path / "feature_group_importance.csv", index=False)

    submission = build_submission(test["ID"], test_probability)
    submission.to_csv(submission_path / f"{run_name}_submission.csv", index=False)
    pd.DataFrame({"ID": train["ID"], "y_true": y, "oof_probability": oof, "oof_label_at_0_5": (oof >= 0.5).astype(int)}).to_csv(table_path / "oof_predictions.csv", index=False)

    # Fit one full-data model for a portable artifact; submission is the more stable fold ensemble above.
    final_model = make_model(SEEDS[0], params=params)
    final_model.fit(X, y, cat_features=cat_indices)
    final_model.save_model(str(model_path / "catboost_final.cbm"))
    joblib.dump({"features": X.columns.tolist(), "categorical": categorical, "numeric": numeric}, model_path / "feature_schema.joblib")

    with (report_path / f"{run_name}_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    if run_name == "catboost_all_features":
        with (report_path / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(overall, handle, indent=2)
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--all-features", action="store_true", help="Ignore models/selected_features.json and train on every engineered feature")
    parser.add_argument("--run-name", default="catboost_ensemble")
    args = parser.parse_args()
    print(json.dumps(run_training(args.data_dir, args.report_dir, args.submission_dir, args.model_dir, use_selected=not args.all_features, run_name=args.run_name), indent=2))
