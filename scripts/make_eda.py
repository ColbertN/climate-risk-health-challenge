"""Create the public-facing exploratory analysis artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from climate_health.features import TARGET, load_competition_data


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(data_dir: str | Path = ".", report_dir: str | Path = "reports") -> pd.DataFrame:
    train, test, climate = load_competition_data(data_dir)
    out = Path(report_dir)
    fig_dir = out / "figures"
    table_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    train["deathdate"] = pd.to_datetime(train["deathdate"])
    test["deathdate"] = pd.to_datetime(test["deathdate"])
    merged = train.merge(climate.drop(columns=["deathdate"]), on="ID", how="left", validate="one_to_one")
    merged["month"] = merged["deathdate"].dt.month
    merged["year"] = merged["deathdate"].dt.year
    merged["age_band"] = pd.cut(
        merged["age"],
        bins=[-np.inf, 1, 5, 14, 24, 44, 64, np.inf],
        labels=["infant", "young child", "child", "young adult", "adult", "older adult", "elderly"],
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train[TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#2878b5")
    ax.set(title="Target balance", xlabel="Climate-sensitive target", ylabel="Records")
    for i, value in enumerate(counts.values):
        ax.text(i, value + counts.max() * 0.02, f"{value:,}\n({value / len(train):.1%})", ha="center")
    _save(fig, fig_dir / "01_target_balance.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    month_rate = merged.groupby("month", observed=True)[TARGET].mean().reindex(range(1, 13))
    sns.lineplot(x=month_rate.index, y=month_rate.values, marker="o", ax=axes[0], color="#d95f02")
    axes[0].set(title="Climate-sensitive rate by month", xlabel="Month", ylabel="Positive rate")
    year_rate = merged.groupby("year", observed=True)[TARGET].mean()
    sns.lineplot(x=year_rate.index, y=year_rate.values, marker="o", ax=axes[1], color="#1b9e77")
    axes[1].set(title="Climate-sensitive rate by year", xlabel="Year", ylabel="Positive rate")
    _save(fig, fig_dir / "02_temporal_target_rates.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=merged, x="age_band", y=TARGET, ax=axes[0], color="#7570b3", errorbar=None)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set(title="Rate by age band", xlabel="", ylabel="Positive rate")
    sns.barplot(data=merged, x="gender", y=TARGET, ax=axes[1], color="#e7298a", errorbar=None)
    axes[1].set(title="Rate by gender", xlabel="", ylabel="Positive rate")
    sns.barplot(data=merged, x="zone", y=TARGET, ax=axes[2], color="#66a61e", errorbar=None)
    axes[2].set(title="Rate by zone", xlabel="", ylabel="Positive rate")
    _save(fig, fig_dir / "03_demographic_target_rates.png")

    climate_cols = [
        "avg_temperature", "precipitation", "rain_sum_30d", "rain_sum_90d",
        "tavg_30d", "hot_days_30d", "ndvi_30d", "elevation",
    ]
    plot_frame = merged[climate_cols + [TARGET]].melt(id_vars=TARGET, var_name="feature", value_name="value")
    g = sns.FacetGrid(plot_frame, col="feature", col_wrap=4, hue=TARGET, sharex=False, sharey=False, height=2.5)
    g.map_dataframe(sns.histplot, x="value", stat="density", common_norm=False, alpha=0.45, bins=20)
    g.add_legend(title="Target")
    g.set_titles("{col_name}")
    g.figure.suptitle("Environmental feature distributions by target", y=1.02)
    g.figure.savefig(fig_dir / "04_climate_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(g.figure)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=merged, x="longitude", y="latitude", hue=TARGET, size="age", sizes=(15, 100), alpha=0.65, palette={0: "#4575b4", 1: "#d73027"}, ax=ax)
    ax.set(title="Mortality records in geographic space", xlabel="Longitude", ylabel="Latitude")
    _save(fig, fig_dir / "05_spatial_distribution.png")

    numeric = merged.select_dtypes(include=np.number)
    corr = numeric.corr(numeric_only=True)[[TARGET]].sort_values(TARGET, ascending=False)
    top_corr = corr.head(16).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(merged[top_corr].corr(), cmap="vlag", center=0, ax=ax, square=True)
    ax.set(title="Correlations among target-associated numeric features")
    _save(fig, fig_dir / "06_correlation_heatmap.png")

    missing = train.isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#6a3d9a")
    ax.set(title="Missingness in supplied training data", xlabel="Fraction missing", ylabel="")
    _save(fig, fig_dir / "07_missingness.png")

    summary = pd.DataFrame(
        {
            "table": ["Train", "Test", "Climate features"],
            "rows": [len(train), len(test), len(climate)],
            "columns": [train.shape[1], test.shape[1], climate.shape[1]],
        }
    )
    summary.to_csv(table_dir / "dataset_summary.csv", index=False)

    group_rows = []
    for name, col in (("gender", "gender"), ("zone", "zone"), ("district", "location")):
        if col == "location":
            keys = train[col].astype(str).str.split(",").str[1].str.strip().fillna("Unknown")
        else:
            keys = train[col].fillna("Unknown")
        stats = train.assign(_group=keys).groupby("_group")[TARGET].agg(["count", "mean"]).reset_index()
        stats.insert(0, "group_type", name)
        stats.columns = ["group_type", "group", "count", "positive_rate"]
        group_rows.append(stats)
    pd.concat(group_rows, ignore_index=True).to_csv(table_dir / "group_target_rates.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    print(run_eda(args.data_dir, args.report_dir).to_string(index=False))
