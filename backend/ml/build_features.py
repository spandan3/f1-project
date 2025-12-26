# backend/ml/build_features.py
"""
Build feature dataset from raw F1 data.
Unified pipeline for all years - training and prediction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    RAW_DIR,
    PROC_DIR,
    FE_DIR,
    first_col,
    to_seconds,
    compute_pit_losses_sum,
    build_weather_features,
    add_weather_columns,
    add_starting_compound,
    FEATURE_COLS,
    ZERO_FILL_COLS,
)

# Raw data paths
RAW_LAPS = RAW_DIR / "laps.parquet"
RAW_RES = RAW_DIR / "results.parquet"
RAW_WX = RAW_DIR / "weather.parquet"


def build_features(
    output_name: str = "features.parquet",
    min_year: int | None = None,
    max_year: int | None = None,
) -> Path:
    """
    Build feature table from all available raw data.
    
    Args:
        output_name: Name for output parquet file
        min_year: Optional minimum year filter
        max_year: Optional maximum year filter
        
    Returns:
        Path to saved features file
    """
    if not RAW_RES.exists():
        raise FileNotFoundError(
            f"No results data found at {RAW_RES}. "
            "Run: python -m backend.ml.fetch_data --years 2015-2025"
        )

    print("📊 Loading raw data...")
    res = pd.read_parquet(RAW_RES)
    laps = pd.read_parquet(RAW_LAPS) if RAW_LAPS.exists() else pd.DataFrame()
    
    # Apply year filters
    if min_year is not None:
        res = res[res["event_year"] >= min_year]
        if not laps.empty:
            laps = laps[laps["event_year"] >= min_year]
    if max_year is not None:
        res = res[res["event_year"] <= max_year]
        if not laps.empty:
            laps = laps[laps["event_year"] <= max_year]
    
    print(f"   Years: {res['event_year'].min()} - {res['event_year'].max()}")
    print(f"   Results rows: {len(res):,}")
    print(f"   Laps rows: {len(laps):,}")

    # Get column mappings
    st_col = first_col(res, ["session_type", "SessionType"], None)
    if st_col is None:
        raise ValueError("Results data has no session type column")

    drv_col = first_col(res, ["Driver", "Abbreviation", "DriverId"], "Driver")
    team_col = first_col(res, ["TeamName", "Team"], "TeamName")
    grid_col = first_col(res, ["GridPosition"], "GridPosition")
    pos_col = first_col(res, ["Position"], "Position")
    stat_col = first_col(res, ["Status"], "Status")
    year_col = first_col(res, ["event_year", "EventYear"], "event_year")
    name_col = first_col(res, ["event_name", "EventName"], "event_name")
    dnum_col = first_col(res, ["DriverNumber"], "DriverNumber")

    # === QUALIFYING FEATURES ===
    print("🔧 Building qualifying features...")
    quali = res[res[st_col].astype(str) == "Q"].copy()
    for q in ["Q1", "Q2", "Q3"]:
        if q not in quali.columns:
            quali[q] = np.nan

    quali = quali.rename(columns={
        drv_col: "Driver",
        team_col: "TeamName",
        year_col: "event_year",
        name_col: "event_name",
        dnum_col: "DriverNumber",
        pos_col: "QualiPos",
        grid_col: "GridPosition"
    })
    
    quali["q1_s"] = quali["Q1"].map(to_seconds)
    quali["q2_s"] = quali["Q2"].map(to_seconds)
    quali["q3_s"] = quali["Q3"].map(to_seconds)
    quali["best_q"] = quali[["q1_s", "q2_s", "q3_s"]].min(axis=1, skipna=True)
    quali["quali_gap_s"] = quali["best_q"] - quali.groupby(["event_year", "event_name"])["best_q"].transform("min")
    quali["quali_pos_final"] = quali["QualiPos"].where(quali["QualiPos"].notna(), quali["GridPosition"])

    quali_feat = quali[["DriverNumber", "event_year", "event_name", "quali_pos_final", "quali_gap_s"]].copy()

    # === RACE RESULTS ===
    print("🔧 Building race features...")
    race = res[res[st_col].astype(str) == "R"].copy().rename(columns={
        drv_col: "Driver",
        team_col: "TeamName",
        year_col: "event_year",
        name_col: "event_name",
        dnum_col: "DriverNumber",
        pos_col: "finish_pos",
        stat_col: "Status",
        grid_col: "GridPosition"
    })

    # Merge quali into race
    df = pd.merge(
        race[["DriverNumber", "Driver", "TeamName", "event_year", "event_name", 
              "finish_pos", "Status", "GridPosition"]],
        quali_feat,
        on=["DriverNumber", "event_year", "event_name"],
        how="left"
    )

    # Grid position
    df["grid_pos"] = df["GridPosition"].where(df["GridPosition"].notna(), df["quali_pos_final"])
    df["grid_quali_diff"] = df["grid_pos"] - df["quali_pos_final"]
    df = df.drop(columns=["GridPosition"])

    # Sort chronologically for rolling calculations
    df = df.sort_values(["Driver", "event_year", "event_name"], kind="mergesort").reset_index(drop=True)

    # === ROLLING AVERAGES ===
    print("🔧 Computing rolling averages...")
    last3 = lambda s: s.shift().rolling(3, min_periods=1).mean()
    df["driver_last3_avg_finish"] = df.groupby("Driver", group_keys=False)["finish_pos"].transform(last3)
    df["team_last3_avg_finish"] = df.groupby("TeamName", group_keys=False)["finish_pos"].transform(last3)

    # Fill early-season gaps
    for c in ["driver_last3_avg_finish", "team_last3_avg_finish"]:
        key = "Driver" if c.startswith("driver") else "TeamName"
        df[c] = df.groupby(key)[c].transform(lambda s: s.fillna(s.mean()))
        df[c] = df[c].fillna(df[c].mean())

    # === RACE DYNAMICS ===
    df["pos_change"] = df["grid_pos"] - df["finish_pos"]
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)
    
    # Total overtakes per race
    overtakes = (
        df.assign(_gain=df["pos_change"].clip(lower=0))
        .groupby("race_id")["_gain"]
        .sum()
        .rename("race_total_overtakes")
        .reset_index()
    )
    df = df.merge(overtakes, on="race_id", how="left")

    # === PER-DRIVER OVERTAKES ===
    if not laps.empty:
        required_cols = ["event_year", "event_name", "Driver", "LapNumber", "Position"]
        if all(c in laps.columns for c in required_cols):
            print("🔧 Computing per-driver overtakes...")
            laps_sorted = laps.sort_values(["event_year", "event_name", "Driver", "LapNumber"])
            pos_diff = laps_sorted.groupby(["event_year", "event_name", "Driver"])["Position"].diff()

            drv_ov = (pos_diff < 0).groupby(
                [laps_sorted["event_year"], laps_sorted["event_name"], laps_sorted["Driver"]]
            ).sum().reset_index(name="driver_overtakes")

            drv_be = (pos_diff > 0).groupby(
                [laps_sorted["event_year"], laps_sorted["event_name"], laps_sorted["Driver"]]
            ).sum().reset_index(name="driver_times_overtaken")

            drv = drv_ov.merge(drv_be, on=["event_year", "event_name", "Driver"], how="outer").fillna(0)
            drv["driver_net_passes"] = drv["driver_overtakes"] - drv["driver_times_overtaken"]

            # Remove duplicates before merge
            for c in ["driver_overtakes", "driver_times_overtaken", "driver_net_passes"]:
                if c in df.columns:
                    df = df.drop(columns=[c])

            df = df.merge(drv, on=["event_year", "event_name", "Driver"], how="left")
        else:
            df["driver_overtakes"] = np.nan
            df["driver_times_overtaken"] = np.nan
            df["driver_net_passes"] = np.nan
    else:
        df["driver_overtakes"] = np.nan
        df["driver_times_overtaken"] = np.nan
        df["driver_net_passes"] = np.nan

    # === WEATHER ===
    print("🔧 Adding weather features...")
    wx_feat = build_weather_features(RAW_WX)
    df = add_weather_columns(df, wx_feat, drop_humidity=True)

    # === PIT LOSSES ===
    if not laps.empty:
        print("🔧 Computing pit stop losses...")
        pit_tot = compute_pit_losses_sum(laps.copy(), n_baseline=3)
        df = df.merge(pit_tot, on=["event_year", "event_name", "DriverNumber"], how="left")
    else:
        df["pit_loss_total_s"] = np.nan

    # === STARTING COMPOUND ===
    if not laps.empty:
        print("🔧 Adding tyre compound...")
        df = add_starting_compound(df, laps)
    else:
        df["start_compound"] = np.nan
        for comp in ["soft", "medium", "hard", "inter", "wet"]:
            df[f"start_{comp}"] = np.nan

    # === TARGET ===
    df["scored_points"] = (pd.to_numeric(df["finish_pos"], errors="coerce") <= 10).astype(int)

    # === SAVE PROCESSED ===
    proc_path = PROC_DIR / "driver_race_processed.parquet"
    df.to_parquet(proc_path, index=False)

    # === FINAL FEATURE SET ===
    drop_cols = [c for c in ["relevance", "track_overtake_idx", "pit_loss_s"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    keep = ["race_id", "event_year", "event_name", "Driver", "DriverNumber", "TeamName",
            "finish_pos", "scored_points"] + feature_cols
    out = df[keep].copy()

    # NA handling
    for c in ZERO_FILL_COLS:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    if "grid_pos" in out.columns:
        out = out.dropna(subset=["grid_pos"])

    # Save features
    out_path = FE_DIR / output_name
    out.to_parquet(out_path, index=False)
    
    n_races = out["race_id"].nunique()
    n_years = out["event_year"].nunique()
    
    print(f"\n✅ Saved features → {out_path}")
    print(f"   Rows: {len(out):,} | Races: {n_races} | Years: {n_years} | Features: {len(feature_cols)}")
    
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build F1 prediction features")
    parser.add_argument(
        "--output", type=str, default="features.parquet",
        help="Output filename (default: features.parquet)"
    )
    parser.add_argument(
        "--min-year", type=int, default=None,
        help="Minimum year to include"
    )
    parser.add_argument(
        "--max-year", type=int, default=None,
        help="Maximum year to include"
    )
    args = parser.parse_args()
    
    build_features(args.output, args.min_year, args.max_year)

