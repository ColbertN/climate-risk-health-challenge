"""Train a complementary LightGBM model and choose a leakage-safe OOF blend."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climate_health.features import TARGET, build_features, load_competition_data, load_external_features  # noqa: E402
from climate_health.postprocess import build_submission  # noqa: E402


def score(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    f1 = f1_score(y, (probability >= 0.5).astype(int))
    auc = roc_auc_score(y, probability)
    return {"f1_default_0_5": float(f1), "roc_auc": float(auc), "weighted_score": float(0.6 * f1 + 0.4 * auc)}


def prepare_matrix(X: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    both = pd.concat([X, X_test], ignore_index=True)
    both = pd.get_dummies(both, columns=both.select_dtypes(include=["object", "string", "category"]).columns.tolist(), dummy_na=True)
    both.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(col)) for col in both.columns]
    train_matrix = both.iloc[: len(X)].copy()
    test_matrix = both.iloc[len(X) :].copy()
    medians = train_matrix.median(numeric_only=True)
    train_matrix = train_matrix.fillna(medians).fillna(0).astype(float)
    test_matrix = test_matrix.fillna(medians).fillna(0).astype(float)
    return train_matrix, test_matrix


def run_blend(data_dir: str | Path = ".", report_dir: str | Path = "reports", submission_dir: str | Path = "submissions") -> dict:
    train, test, climate = load_competition_data(data_dir)
    external = load_external_features(data_dir)
    y = train[TARGET].to_numpy(dtype=int)
    raw_train = train.drop(columns=[TARGET])
    combined = pd.concat([raw_train, test], ignore_index=True)
    X_cat = build_features(raw_train, climate, combined_for_counts=combined, external_features=external)
    X_cat_test = build_features(test, climate, combined_for_counts=combined, external_features=external)
    X, X_test = prepare_matrix(X_cat, X_cat_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(train), dtype=float)
    test_probability = np.zeros(len(test), dtype=float)
    xgb_oof = np.zeros(len(train), dtype=float)
    xgb_test_probability = np.zeros(len(test), dtype=float)
    importance_rows = []
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.025,
            num_leaves=24,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=8.0,
            objective="binary",
            random_state=2026 + fold,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(
            X.iloc[fit_idx], y[fit_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        test_probability += model.predict_proba(X_test)[:, 1] / cv.n_splits
        importance_rows.append(pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_, "fold": fold}))

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=10.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=2026 + fold,
            n_jobs=4,
        )
        xgb.fit(X.iloc[fit_idx], y[fit_idx], eval_set=[(X.iloc[valid_idx], y[valid_idx])], verbose=False)
        xgb_oof[valid_idx] = xgb.predict_proba(X.iloc[valid_idx])[:, 1]
        xgb_test_probability += xgb.predict_proba(X_test)[:, 1] / cv.n_splits

    cat_oof_path = Path(report_dir) / "tables" / "oof_predictions.csv"
    cat_oof = pd.read_csv(cat_oof_path)
    cat_probability = cat_oof.set_index("ID").loc[train["ID"], "oof_probability"].to_numpy()
    cat_test = pd.read_csv(Path(submission_dir) / "nasa_power_all_features_submission.csv")
    cat_test_probability = cat_test.set_index("ID").loc[test["ID"], "TargetRAUC"].to_numpy()

    blend_rows = []
    for cat_weight in np.linspace(0, 1, 21):
        for lgb_weight in np.linspace(0, 1 - cat_weight, 21):
            xgb_weight = 1 - cat_weight - lgb_weight
            probability = cat_weight * cat_probability + lgb_weight * oof + xgb_weight * xgb_oof
            blend_rows.append({"catboost_weight": float(cat_weight), "lightgbm_weight": float(lgb_weight), "xgboost_weight": float(xgb_weight), **score(y, probability)})
    blend_results = pd.DataFrame(blend_rows).sort_values("weighted_score", ascending=False).reset_index(drop=True)
    best = blend_results.iloc[0]
    final_probability = float(best["catboost_weight"]) * cat_test_probability + float(best["lightgbm_weight"]) * test_probability + float(best["xgboost_weight"]) * xgb_test_probability

    report_path = Path(report_dir) / "tables"
    report_path.mkdir(parents=True, exist_ok=True)
    blend_results.to_csv(report_path / "blend_search.csv", index=False)
    pd.DataFrame({"ID": train["ID"], "y_true": y, "lightgbm_oof_probability": oof, "xgboost_oof_probability": xgb_oof, "catboost_oof_probability": cat_probability}).to_csv(report_path / "blend_oof_predictions.csv", index=False)
    pd.concat(importance_rows, ignore_index=True).groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).to_csv(report_path / "lightgbm_feature_importance.csv", index=False)
    submission = build_submission(test["ID"], final_probability)
    Path(submission_dir).mkdir(parents=True, exist_ok=True)
    submission.to_csv(Path(submission_dir) / "nasa_power_catboost_lgbm_blend_submission.csv", index=False)
    summary = {"lightgbm": score(y, oof), "xgboost": score(y, xgb_oof), "catboost": score(y, cat_probability), "blend": score(y, cat_probability * float(best["catboost_weight"]) + oof * float(best["lightgbm_weight"]) + xgb_oof * float(best["xgboost_weight"])), "best_catboost_weight": float(best["catboost_weight"]), "best_lightgbm_weight": float(best["lightgbm_weight"]), "best_xgboost_weight": float(best["xgboost_weight"]), "n_features_after_one_hot": int(X.shape[1])}
    with (Path(report_dir) / "blend_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--submission-dir", default="submissions")
    args = parser.parse_args()
    print(json.dumps(run_blend(args.data_dir, args.report_dir, args.submission_dir), indent=2))
