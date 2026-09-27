"""Validate a challenge submission against Test.csv and the official schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(data_dir: str | Path = ".", submission_path: str | Path = "submissions/catboost_all_features_submission.csv", report_dir: str | Path = "reports") -> dict:
    test = pd.read_csv(Path(data_dir) / "Test.csv")
    submission = pd.read_csv(submission_path)
    expected = ["ID", "TargetF1", "TargetRAUC"]
    checks = {
        "columns_exact": submission.columns.tolist() == expected,
        "row_count_matches_test": len(submission) == len(test),
        "ids_match_test_order": submission["ID"].tolist() == test["ID"].tolist(),
        "ids_unique": not submission["ID"].duplicated().any(),
        "target_f1_binary": set(submission["TargetF1"].unique()).issubset({0, 1}),
        "probabilities_finite": bool(np.isfinite(submission["TargetRAUC"]).all()),
        "probabilities_in_unit_interval": bool(submission["TargetRAUC"].between(0, 1).all()),
        "labels_match_default_threshold": bool((submission["TargetF1"].to_numpy() == (submission["TargetRAUC"].to_numpy() >= 0.5)).all()),
    }
    result = {"passed": bool(all(checks.values())), "checks": checks, "rows": int(len(submission)), "positive_labels": int(submission["TargetF1"].sum()), "probability_min": float(submission["TargetRAUC"].min()), "probability_max": float(submission["TargetRAUC"].max())}
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with (Path(report_dir) / "submission_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if not result["passed"]:
        raise ValueError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--submission", default="submissions/catboost_all_features_submission.csv")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir, args.submission, args.report_dir), indent=2))
