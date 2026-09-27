"""Small explicit CatBoost search; intentionally not an AutoML system."""

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
from climate_health.features import TARGET, build_features, load_competition_data, split_feature_types  # noqa: E402


SEARCH_SPACE = [
    {"name": "balanced_depth6", "depth": 6, "learning_rate": 0.035, "l2_leaf_reg": 6.0, "random_strength": 0.5},
    {"name": "regularized_depth7", "depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 10.0, "random_strength": 0.7},
]


def run_search(data_dir: str | Path = ".", report_dir: str | Path = "reports", model_dir: str | Path = "models") -> pd.DataFrame:
    train, test, climate = load_competition_data(data_dir)
    y = train[TARGET].to_numpy(dtype=int)
    raw_train = train.drop(columns=[TARGET])
    combined = pd.concat([raw_train, test], ignore_index=True)
    X = build_features(raw_train, climate, combined_for_counts=combined)
    numeric, categorical = split_feature_types(X)
    cat_indices = [X.columns.get_loc(col) for col in categorical]
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=2026)
    rows = []
    for params in SEARCH_SPACE:
        fold_scores = []
        for fit_idx, valid_idx in cv.split(X, y):
            model = CatBoostClassifier(
                iterations=250,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_strength=params["random_strength"],
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
            fold_scores.append((f1, auc, 0.6 * f1 + 0.4 * auc))
        mean = np.mean(fold_scores, axis=0)
        rows.append({"name": params["name"], "f1_default_0_5": mean[0], "roc_auc": mean[1], "weighted_score": mean[2], **{k: v for k, v in params.items() if k != "name"}})
    results = pd.DataFrame(rows).sort_values("weighted_score", ascending=False).reset_index(drop=True)
    (Path(report_dir) / "tables").mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(report_dir) / "tables" / "hyperparameter_search.csv", index=False)
    best = results.iloc[0].to_dict()
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with (Path(model_dir) / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    print(run_search(args.data_dir, args.report_dir, args.model_dir).to_string(index=False))
