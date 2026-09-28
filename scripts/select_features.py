"""Evaluate importance-ranked feature subsets with an explicit CV experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climate_health.features import TARGET, build_features, load_competition_data, load_external_features, split_feature_types  # noqa: E402


def run_selection(data_dir: str | Path = ".", report_dir: str | Path = "reports", model_dir: str | Path = "models") -> pd.DataFrame:
    train, test, climate = load_competition_data(data_dir)
    external = load_external_features(data_dir)
    importance_path = Path(report_dir) / "tables" / "feature_importance.csv"
    if not importance_path.exists():
        raise FileNotFoundError("Run train_model.py once before select_features.py")
    importance = pd.read_csv(importance_path).sort_values("importance", ascending=False)
    y = train[TARGET].to_numpy(dtype=int)
    raw_train = train.drop(columns=[TARGET])
    combined = pd.concat([raw_train, test], ignore_index=True)
    X_all = build_features(raw_train, climate, combined_for_counts=combined, external_features=external)
    available = [feature for feature in importance["feature"] if feature in X_all.columns]
    candidate_sizes = sorted(set([min(20, len(available)), min(30, len(available)), min(40, len(available)), len(available)]))
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=2026)
    rows = []
    subset_map = {}
    for size in candidate_sizes:
        selected = available[:size]
        subset_map[str(size)] = selected
        X = X_all[selected]
        _, categorical = split_feature_types(X)
        cat_indices = [X.columns.get_loc(col) for col in categorical]
        scores = []
        for fit_idx, valid_idx in cv.split(X, y):
            model = CatBoostClassifier(
                iterations=250,
                depth=6,
                learning_rate=0.035,
                l2_leaf_reg=6.0,
                random_strength=0.5,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=17,
                verbose=False,
                allow_writing_files=False,
                thread_count=4,
            )
            model.fit(X.iloc[fit_idx], y[fit_idx], cat_features=cat_indices)
            probability = model.predict_proba(X.iloc[valid_idx])[:, 1]
            f1 = f1_score(y[valid_idx], (probability >= 0.5).astype(int))
            auc = roc_auc_score(y[valid_idx], probability)
            scores.append((f1, auc, 0.6 * f1 + 0.4 * auc))
        mean = np.mean(scores, axis=0)
        rows.append({"n_features": size, "f1_default_0_5": mean[0], "roc_auc": mean[1], "weighted_score": mean[2]})
    results = pd.DataFrame(rows).sort_values("weighted_score", ascending=False).reset_index(drop=True)
    best_size = str(int(results.iloc[0]["n_features"]))
    (Path(report_dir) / "tables").mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(report_dir) / "tables" / "feature_selection_cv.csv", index=False)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with (Path(model_dir) / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump({"n_features": int(best_size), "features": subset_map[best_size]}, handle, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    print(run_selection(args.data_dir, args.report_dir, args.model_dir).to_string(index=False))
