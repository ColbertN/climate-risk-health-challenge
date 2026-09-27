"""Create a compact feature-importance graphic for the README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_plot(report_dir: str | Path = "reports") -> None:
    root = Path(report_dir)
    importance = pd.read_csv(root / "tables" / "feature_importance.csv").head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#2c7fb8", ax=ax)
    ax.set(title="Top 15 mean CatBoost feature importances", xlabel="Mean importance", ylabel="")
    fig.tight_layout()
    fig.savefig(root / "figures" / "08_feature_importance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    make_plot(args.report_dir)
