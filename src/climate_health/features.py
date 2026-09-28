"""Leakage-safe feature engineering for the Climate Risk and Health challenge."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET = "is_climate_sensitive"
ID_COL = "ID"
DATE_COL = "deathdate"


def load_competition_data(data_dir: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the supplied competition files and verify their expected keys."""

    root = Path(data_dir)
    train = pd.read_csv(root / "Train.csv")
    test = pd.read_csv(root / "Test.csv")
    climate = pd.read_csv(root / "climate_features.csv")

    for name, frame in (("train", train), ("test", test), ("climate_features", climate)):
        if ID_COL not in frame:
            raise ValueError(f"{name} is missing required key column {ID_COL!r}")
        if frame[ID_COL].duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs")

    if set(train[ID_COL]).intersection(test[ID_COL]):
        raise ValueError("Train and Test contain overlapping IDs")
    if not set(train[ID_COL]).issubset(set(climate[ID_COL])) or not set(test[ID_COL]).issubset(set(climate[ID_COL])):
        raise ValueError("climate_features.csv does not cover every Train/Test ID")

    return train, test, climate


def load_external_features(data_dir: str | Path = ".") -> pd.DataFrame | None:
    """Load cached climate-only NASA POWER features when they have been downloaded."""

    path = Path(data_dir) / "data" / "external" / "nasa_power_features.csv"
    if not path.exists():
        return None
    external = pd.read_csv(path)
    if ID_COL not in external or external[ID_COL].duplicated().any():
        raise ValueError("NASA POWER feature cache must have unique ID values")
    return external


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _split_location(value: object) -> tuple[str, str, str]:
    """Extract stable geographic text fields without using target information."""

    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    primary = parts[0] if parts else "Unknown"
    district = parts[1] if len(parts) > 1 else "Unknown"
    region = parts[2] if len(parts) > 2 else district
    return primary, district, region


def build_features(
    frame: pd.DataFrame,
    climate: pd.DataFrame,
    combined_for_counts: pd.DataFrame | None = None,
    external_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create model-ready features.

    All features are derived from a row's covariates, the supplied environmental
    observations, or unsupervised frequencies computed from train+test. No target
    aggregation is used, so this function can safely be applied before splitting.
    """

    base = frame.copy()
    climate_only = climate.drop(columns=[DATE_COL], errors="ignore")
    data = base.merge(climate_only, on=ID_COL, how="left", validate="one_to_one")
    if external_features is not None:
        external_only = external_features.drop(columns=[DATE_COL], errors="ignore")
        data = data.merge(external_only, on=ID_COL, how="left", validate="one_to_one")
    date = pd.to_datetime(data[DATE_COL], errors="coerce")

    data["year"] = date.dt.year.astype("float64")
    data["month"] = date.dt.month.astype("float64")
    data["day_of_year"] = date.dt.dayofyear.astype("float64")
    data["week_of_year"] = date.dt.isocalendar().week.astype("float64")
    data["quarter"] = date.dt.quarter.astype("float64")
    data["season_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
    data["season_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)

    data["age_missing"] = data["age"].isna().astype("int8")
    data["age_sq"] = data["age"].clip(lower=0).pow(2)
    data["is_infant"] = (data["age"] <= 1).astype("int8")
    data["is_child"] = data["age"].between(2, 14, inclusive="both").astype("int8")
    data["is_working_age"] = data["age"].between(15, 64, inclusive="both").astype("int8")
    data["is_older_adult"] = (data["age"] >= 65).astype("int8")
    data["age_band"] = pd.cut(
        data["age"],
        bins=[-np.inf, 1, 5, 14, 24, 44, 64, np.inf],
        labels=["infant", "young_child", "child", "young_adult", "adult", "older_adult", "elderly"],
    ).astype("string").fillna("Unknown")

    for col in ("location", "zone", "gender"):
        data[col] = data[col].astype("string").fillna("Unknown")
    location_parts = data["location"].map(_split_location)
    data["location_primary"] = location_parts.map(lambda x: x[0]).astype("string")
    data["district"] = location_parts.map(lambda x: x[1]).astype("string")
    data["region"] = location_parts.map(lambda x: x[2]).astype("string")

    # Local climate contrasts and rolling-window ratios from CHIRPS/ERA5-Land.
    data["diurnal_temp_range"] = data["max_temperature"] - data["min_temperature"]
    data["daily_temp_center_error"] = data["avg_temperature"] - (
        data["max_temperature"] + data["min_temperature"]
    ) / 2
    data["rain_log"] = np.log1p(data["precipitation"].clip(lower=0))
    data["rain_7_to_30"] = _safe_ratio(data["rain_sum_7d"], data["rain_sum_30d"])
    data["rain_30_to_90"] = _safe_ratio(data["rain_sum_30d"], data["rain_sum_90d"])
    data["rain_intensity_30d"] = _safe_ratio(data["rain_sum_30d"], data["rain_days_30d"])
    data["rain_extreme_share"] = _safe_ratio(data["max_daily_rain_30d"], data["rain_sum_30d"])
    data["temp_7_vs_30"] = data["tavg_7d"] - data["tavg_30d"]
    data["temp_90_vs_30"] = data["tavg_90d"] - data["tavg_30d"]
    data["tmax_tmin_window"] = data["tmax_30d"] - data["tmin_30d"]
    data["hot_day_share"] = data["hot_days_30d"] / 30.0
    data["ndvi_change"] = data["ndvi_30d"] - data["ndvi_90d"]
    data["heat_stress_proxy"] = data["tmax_30d"] * (1 + data["hot_day_share"])
    data["wet_heat_proxy"] = data["tavg_30d"] * (1 + data["rain_days_30d"] / 30.0)

    # Independent NASA POWER weather summaries and cross-source contrasts.
    if "power_t2m_mean_30d" in data:
        data["power_temp_7_vs_30"] = data["power_t2m_mean_7d"] - data["power_t2m_mean_30d"]
        data["power_temp_90_vs_30"] = data["power_t2m_mean_90d"] - data["power_t2m_mean_30d"]
        data["power_rh_7_vs_30"] = data["power_rh_mean_7d"] - data["power_rh_mean_30d"]
        data["power_wind_7_vs_30"] = data["power_wind_mean_7d"] - data["power_wind_mean_30d"]
        data["power_rain_30_to_90"] = _safe_ratio(data["power_rain_sum_30d"], data["power_rain_sum_90d"])
        data["power_rain_log_30d"] = np.log1p(data["power_rain_sum_30d"].clip(lower=0))
        data["power_moisture_deficit_30d"] = data["power_t2m_mean_30d"] - data["power_dewpoint_mean_30d"]
        data["power_heat_humidity_proxy"] = data["power_t2m_mean_30d"] * (1 + data["power_rh_mean_30d"] / 100.0)
        data["power_solar_temp_30d"] = data["power_solar_mean_30d"] * data["power_t2m_mean_30d"]
        data["power_tmax_tmin_range_30d"] = data["power_tmax_max_30d"] - data["power_tmin_max_30d"]

    # Smooth spatial coordinates retain broad geography while avoiding high-cardinality
    # exact-coordinate memorisation.
    data["lat_sq"] = data["latitude"].pow(2)
    data["lon_sq"] = data["longitude"].pow(2)
    data["lat_lon_interaction"] = data["latitude"] * data["longitude"]
    data["spatial_cell"] = (
        data["latitude"].round(2).astype("string") + "_" + data["longitude"].round(2).astype("string")
    )

    # Unsupervised support counts are computed from both covariate tables only.
    if combined_for_counts is None:
        combined_for_counts = frame
    count_source = combined_for_counts.copy()
    count_source["location"] = count_source["location"].astype("string").fillna("Unknown")
    count_source["date_key"] = pd.to_datetime(count_source[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
    loc_counts = count_source["location"].value_counts(dropna=False)
    date_counts = count_source["date_key"].value_counts(dropna=False)
    data["location_support"] = data["location"].map(loc_counts).fillna(0).astype("float64")
    data["date_support"] = date_counts.reindex(date.dt.strftime("%Y-%m-%d")).fillna(0).to_numpy()

    data = data.drop(columns=[ID_COL, DATE_COL], errors="ignore")
    for col in data.select_dtypes(include=["object", "string", "category"]).columns:
        data[col] = data[col].astype("string").fillna("Unknown")
    return data


def split_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns for CatBoost."""

    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [col for col in features.columns if col not in categorical]
    return numeric, categorical


def feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Group features for readable feature-importance reporting."""

    groups = {"demographic": [], "geographic": [], "climate": [], "temporal": [], "support": []}
    for col in columns:
        if col in {"age", "age_sq", "age_missing", "is_infant", "is_child", "is_working_age", "is_older_adult", "age_band", "gender", "zone"}:
            groups["demographic"].append(col)
        elif any(token in col for token in ("lat", "lon", "location", "district", "region", "spatial", "elevation", "slope")):
            groups["geographic"].append(col)
        elif any(token in col for token in ("year", "month", "day", "week", "quarter", "season")):
            groups["temporal"].append(col)
        elif "support" in col:
            groups["support"].append(col)
        else:
            groups["climate"].append(col)
    return groups
