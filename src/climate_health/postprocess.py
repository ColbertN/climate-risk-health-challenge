"""Competition-safe submission post-processing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_submission(ids: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Validate raw probabilities and derive labels at the mandated 0.5 threshold."""

    probs = np.asarray(probabilities, dtype=float)
    if len(ids) != len(probs):
        raise ValueError("IDs and probabilities have different lengths")
    if not np.isfinite(probs).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values")
    # Only guard floating-point spillover. No probability rounding or threshold tuning.
    probs = np.clip(probs, 0.0, 1.0)
    submission = pd.DataFrame(
        {
            "ID": ids.to_numpy(),
            "TargetF1": (probs >= 0.5).astype(int),
            "TargetRAUC": probs,
        }
    )
    if submission["ID"].duplicated().any():
        raise ValueError("Submission IDs are not unique")
    if not bool(submission["TargetF1"].isin([0, 1]).all()):
        raise ValueError("TargetF1 must be binary")
    if not ((submission["TargetRAUC"] >= 0) & (submission["TargetRAUC"] <= 1)).all():
        raise ValueError("TargetRAUC must be in [0, 1]")
    return submission
