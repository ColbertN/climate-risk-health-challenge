"""Leakage-aware smoothed target encoding for geography/time groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from blend_models import prepare_matrix  # noqa: E402
from climate_health.features import TARGET, build_features, load_competition_data, load_external_features  # noqa: E402
from climate_health.postprocess import build_submission  # noqa: E402


KEYS = ["location", "district", "region", "spatial_cell", "age_band", "gender", "zone", "month", "year"]
COMBINATIONS = [("location", "month"), ("district", "month"), ("age_band", "month"), ("spatial_cell", "month"), ("location", "age_band")]


def score(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    f1 = f1_score(y, (probability >= 0.5).astype(int))
    auc = roc_auc_score(y, probability)
    return {"f1_default_0_5": float(f1), "roc_auc": float(auc), "weighted_score": float(0.6 * f1 + 0.4 * auc)}


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for left, right in COMBINATIONS:
        result[f"{left}__{right}"] = result[left].astype(str) + "__" + result[right].astype(str)
    return result


def target_encode(fit_frame: pd.DataFrame, fit_y: np.ndarray, query_frame: pd.DataFrame, smoothing: float = 25.0) -> pd.DataFrame:
    fit_frame = add_keys(fit_frame)
    query_frame = add_keys(query_frame)
    prior = float(np.mean(fit_y))
    output = pd.DataFrame(index=query_frame.index)
    for key in KEYS + [f"{left}__{right}" for left, right in COMBINATIONS]:
        fit_key = fit_frame[key].astype(str)
        stats = pd.DataFrame({"key": fit_key.to_numpy(), "target": fit_y}).groupby("key")["target"].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
        output[f"te_{key}"] = query_frame[key].astype(str).map(smoothed).fillna(prior).astype(float).to_numpy()
    return output.reset_index(drop=True)


def run(data_dir: str | Path = ".", report_dir: str | Path = "reports", submission_dir: str | Path = "submissions") -> dict:
    train, test, climate = load_competition_data(data_dir)
    external = load_external_features(data_dir)
    y = train[TARGET].to_numpy(dtype=int)
    raw_train = train.drop(columns=[TARGET])
    combined = pd.concat([raw_train, test], ignore_index=True)
    engineered = build_features(raw_train, climate, combined_for_counts=combined, external_features=external)
    engineered_test = build_features(test, climate, combined_for_counts=combined, external_features=external)
    base, base_test = prepare_matrix(engineered, engineered_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(base, y), start=1):
        fit_te = target_encode(engineered.iloc[fit_idx], y[fit_idx], engineered.iloc[fit_idx])
        valid_te = target_encode(engineered.iloc[fit_idx], y[fit_idx], engineered.iloc[valid_idx])
        test_te = target_encode(engineered.iloc[fit_idx], y[fit_idx], engineered_test)
        fit_matrix = pd.concat([base.iloc[fit_idx].reset_index(drop=True), fit_te], axis=1)
        valid_matrix = pd.concat([base.iloc[valid_idx].reset_index(drop=True), valid_te], axis=1)
        test_matrix = pd.concat([base_test.reset_index(drop=True), test_te], axis=1)
        model = LGBMClassifier(
            n_estimators=650,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=10.0,
            objective="binary",
            random_state=2026 + fold,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(fit_matrix, y[fit_idx], eval_set=[(valid_matrix, y[valid_idx])], callbacks=[early_stopping(80, verbose=False), log_evaluation(0)])
        oof[valid_idx] = model.predict_proba(valid_matrix)[:, 1]
        test_pred += model.predict_proba(test_matrix)[:, 1] / cv.n_splits

    report = score(y, oof)
    report["n_features"] = int(base.shape[1] + len(KEYS) + len(COMBINATIONS))
    Path(report_dir, "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": train["ID"], "y_true": y, "target_encoded_oof_probability": oof}).to_csv(Path(report_dir) / "tables" / "target_encoded_oof.csv", index=False)
    submission = build_submission(test["ID"], test_pred)
    Path(submission_dir).mkdir(parents=True, exist_ok=True)
    submission.to_csv(Path(submission_dir) / "target_encoded_lgbm_submission.csv", index=False)
    with (Path(report_dir) / "target_encoded_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--submission-dir", default="submissions")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.report_dir, args.submission_dir), indent=2))
