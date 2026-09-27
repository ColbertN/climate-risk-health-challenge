"""Run EDA and model training from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from make_eda import run_eda  # noqa: E402
from train_model import run_training  # noqa: E402
from tune_hyperparameters import run_search  # noqa: E402
from select_features import run_selection  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT))
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--submission-dir", default=str(ROOT / "submissions"))
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    args = parser.parse_args()
    print("Creating EDA artifacts...")
    run_eda(args.data_dir, args.report_dir)
    print("Running explicit hyperparameter search...")
    run_search(args.data_dir, args.report_dir, args.model_dir)
    print("Training all-feature cross-validated CatBoost ensemble...")
    result = run_training(args.data_dir, args.report_dir, args.submission_dir, args.model_dir)
    print("Selecting an importance-ranked feature subset...")
    run_selection(args.data_dir, args.report_dir, args.model_dir)
    print("Training final selected-feature ensemble...")
    result = run_training(args.data_dir, args.report_dir, args.submission_dir, args.model_dir)
    print(result)
