"""Download and aggregate NASA POWER daily weather features for this challenge.

The API requests are made once per unique 0.5-degree grid cell covering the
provided records. Aggregations use only days strictly before each death date.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = [
    "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "PRECTOTCORR", "RH2M",
    "WS10M", "ALLSKY_SFC_SW_DWN", "EVPTRNS", "GWETTOP", "GWETROOT",
]


def _grid(value: pd.Series) -> pd.Series:
    return (value * 2).round() / 2


def _fetch_grid(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": f"{lon:.2f}",
        "latitude": f"{lat:.2f}",
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = requests.get(API_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    parameter_data = payload["properties"]["parameter"]
    dates = sorted({date for series in parameter_data.values() for date in series})
    result = pd.DataFrame({"date": pd.to_datetime(dates, format="%Y%m%d")})
    for name, series in parameter_data.items():
        result[name.lower()] = pd.Series(series, dtype="float64").reindex(dates).to_numpy()
    result["lat_grid"] = lat
    result["lon_grid"] = lon
    return result


def _aggregate_one(row: pd.Series, daily: pd.DataFrame) -> dict[str, float]:
    before = daily[(daily["date"] < row["death_date"]) & (daily["lat_grid"] == row["lat_grid"]) & (daily["lon_grid"] == row["lon_grid"])].tail(90)
    result: dict[str, float] = {}
    for days in (7, 30, 90):
        window = before.tail(days)
        for source, prefix in (("t2m", "power_t2m"), ("t2mdew", "power_dewpoint"), ("rh2m", "power_rh"), ("ws10m", "power_wind"), ("allsky_sfc_sw_dwn", "power_solar"), ("gwettop", "power_soilwet_top"), ("gwetroot", "power_soilwet_root")):
            result[f"{prefix}_mean_{days}d"] = float(window[source].mean()) if len(window) else np.nan
        for source, prefix in (("t2m", "power_t2m"), ("t2m_max", "power_tmax"), ("t2m_min", "power_tmin")):
            result[f"{prefix}_max_{days}d"] = float(window[source].max()) if len(window) else np.nan
        for source, prefix in (("prectotcorr", "power_rain"), ("evptrns", "power_evap")):
            result[f"{prefix}_sum_{days}d"] = float(window[source].sum()) if len(window) else np.nan
    result["power_temp_range_30d"] = result["power_tmax_max_30d"] - result["power_tmin_max_30d"]
    result["power_rain_7_to_30"] = result["power_rain_sum_7d"] / max(result["power_rain_sum_30d"], 1e-6)
    result["power_rh_temp_30d"] = result["power_rh_mean_30d"] * result["power_t2m_mean_30d"]
    return result


def download(data_dir: str | Path = ".", output: str | Path = "data/external/nasa_power_features.csv") -> pd.DataFrame:
    root = Path(data_dir)
    train = pd.read_csv(root / "Train.csv")
    test = pd.read_csv(root / "Test.csv")
    rows = pd.concat([train.drop(columns=["is_climate_sensitive"], errors="ignore"), test], ignore_index=True)
    rows["death_date"] = pd.to_datetime(rows["deathdate"])
    rows["lat_grid"] = _grid(rows["latitude"])
    rows["lon_grid"] = _grid(rows["longitude"])
    start = (rows["death_date"].min() - pd.Timedelta(days=91)).strftime("%Y%m%d")
    end = rows["death_date"].max().strftime("%Y%m%d")
    daily_frames = []
    for lat, lon in rows[["lat_grid", "lon_grid"]].drop_duplicates().itertuples(index=False):
        print(f"Downloading NASA POWER grid lat={lat}, lon={lon}")
        daily_frames.append(_fetch_grid(float(lat), float(lon), start, end))
        time.sleep(0.5)
    daily = pd.concat(daily_frames, ignore_index=True)
    feature_rows = []
    for _, row in rows.iterrows():
        feature_rows.append({"ID": row["ID"], **_aggregate_one(row, daily)})
    result = pd.DataFrame(feature_rows)
    output_path = root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metadata = {
        "source": "NASA POWER Daily API",
        "source_url": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/point/",
        "parameters": PARAMETERS,
        "grid_resolution_degrees": 0.5,
        "start": start,
        "end": end,
        "rows": len(result),
    }
    with (output_path.parent / "nasa_power_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output", default="data/external/nasa_power_features.csv")
    args = parser.parse_args()
    result = download(args.data_dir, args.output)
    print(result.shape)
