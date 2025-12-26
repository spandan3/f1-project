# backend/ml/common.py
"""
Shared utilities for F1 ML pipeline.
Consolidates duplicated code from build_dataset.py, build_2025_dataset.py, and build_inference_rows.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
# PATH CONSTANTS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
FE_DIR = BASE_DIR / "data" / "fe"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "f1_cache"

# Ensure directories exist
for d in (RAW_DIR, PROC_DIR, FE_DIR, MODEL_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# COLUMN HELPERS
# ============================================================================

def first_col(df: pd.DataFrame, candidates: list[str], default: Optional[str] = None) -> str:
    """
    Find the first existing column from a list of candidates.
    If none found, creates a column with NaN values.
    
    Args:
        df: DataFrame to search
        candidates: List of column names to try in order
        default: Default column name to create if none found
        
    Returns:
        Name of the found or created column
    """
    for c in candidates:
        if c in df.columns:
            return c
    if default:
        if default not in df.columns:
            df[default] = np.nan
        return default
    tmp = f"__missing_{candidates[0]}"
    df[tmp] = np.nan
    return tmp


def to_seconds(x) -> float:
    """
    Convert various time formats to seconds.
    Handles: Timedelta, "M:SS.mmm" strings, numeric values.
    
    Args:
        x: Time value in various formats
        
    Returns:
        Time in seconds as float, or NaN if conversion fails
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, pd.Timedelta):
        return x.total_seconds()
    td = pd.to_timedelta(str(x), errors="coerce")
    if not pd.isna(td):
        return td.total_seconds()
    try:
        m, s = str(x).split(":")
        return int(m) * 60 + float(s)
    except Exception:
        return np.nan


# ============================================================================
# PIT STOP ANALYSIS
# ============================================================================

def compute_pit_losses_sum(laps: pd.DataFrame, n_baseline: int = 3) -> pd.DataFrame:
    """
    Calculate total pit stop time loss per driver per event.
    
    pit_loss = (in_lap_time + out_lap_time) - 2 * baseline_lap_time
    baseline_lap_time = median of previous n clean laps (no pit in/out)
    
    Args:
        laps: DataFrame with lap data including LapTime, PitInTime, PitOutTime
        n_baseline: Number of clean laps to use for baseline calculation
        
    Returns:
        DataFrame with columns: event_year, event_name, DriverNumber, pit_loss_total_s
    """
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["event_year", "event_name", "DriverNumber", "pit_loss_total_s"])

    # Ensure required columns exist
    for c in ["LapNumber", "LapTime", "PitInTime", "PitOutTime", "event_year", "event_name", "DriverNumber", "Driver"]:
        if c not in laps.columns:
            laps[c] = np.nan

    def _sec(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, pd.Timedelta):
            return x.total_seconds()
        td = pd.to_timedelta(x, errors="coerce")
        if not pd.isna(td):
            return td.total_seconds()
        try:
            return float(x)
        except Exception:
            return np.nan

    rows = []
    laps = laps.sort_values(["event_year", "event_name", "Driver", "LapNumber"])
    
    for (ey, en, drv, dnum), g in laps.groupby(["event_year", "event_name", "Driver", "DriverNumber"], sort=False):
        g = g.sort_values("LapNumber")
        out_idx = g.index[g["PitOutTime"].notna()].tolist()
        
        for oi in out_idx:
            out_row = g.loc[oi]
            out_lap = int(out_row["LapNumber"])
            prev_row = g[g["LapNumber"] == out_lap - 1]
            if prev_row.empty:
                continue
            in_row = prev_row.iloc[0]

            # Find clean laps for baseline
            prev_clean = g[
                (g["LapNumber"] < out_lap) & 
                g["PitInTime"].isna() & 
                g["PitOutTime"].isna() & 
                g["LapTime"].notna()
            ].tail(n_baseline)
            
            if len(prev_clean):
                baseline = _sec(prev_clean["LapTime"].median())
            else:
                baseline = _sec(g[g["LapTime"].notna()]["LapTime"].median())

            in_lt = _sec(in_row["LapTime"])
            out_lt = _sec(out_row["LapTime"])
            
            if any(np.isnan([baseline, in_lt, out_lt])):
                loss = np.nan
            else:
                loss = in_lt + out_lt - 2 * baseline
                
            rows.append({
                "event_year": ey,
                "event_name": en,
                "DriverNumber": dnum,
                "pit_loss_s": loss
            })

    per_stop = pd.DataFrame(rows)
    if per_stop.empty:
        return pd.DataFrame(columns=["event_year", "event_name", "DriverNumber", "pit_loss_total_s"])

    return (
        per_stop
        .groupby(["event_year", "event_name", "DriverNumber"])["pit_loss_s"]
        .sum()
        .reset_index()
        .rename(columns={"pit_loss_s": "pit_loss_total_s"})
    )


# ============================================================================
# WEATHER FEATURES
# ============================================================================

def build_weather_features(raw_wx_path: Path) -> Optional[pd.DataFrame]:
    """
    Aggregate weather data for race sessions.
    
    Args:
        raw_wx_path: Path to weather parquet file
        
    Returns:
        DataFrame with aggregated weather features per race, or None if unavailable
    """
    if not raw_wx_path.exists():
        return None
        
    wx = pd.read_parquet(raw_wx_path).copy()
    st_col = first_col(wx, ["session_type", "SessionType"], None)
    if st_col is None:
        return None

    wx_r = wx[wx[st_col].astype(str) == "R"].copy()
    ycol = first_col(wx_r, ["event_year", "EventYear"], "event_year")
    ncol = first_col(wx_r, ["event_name", "EventName"], "event_name")

    agg = {}
    if "AirTemp" in wx_r.columns:
        agg["AirTemp"] = "mean"
    if "TrackTemp" in wx_r.columns:
        agg["TrackTemp"] = "mean"
    if "Humidity" in wx_r.columns:
        agg["Humidity"] = "mean"
    if "Rainfall" in wx_r.columns:
        agg["Rainfall"] = "max"
    if "WindSpeed" in wx_r.columns:
        agg["WindSpeed"] = "mean"
    if "WindDirection" in wx_r.columns:
        agg["WindDirection"] = "mean"
        
    if not agg:
        return None

    wx_feat = (
        wx_r
        .groupby([ycol, ncol])
        .agg(agg)
        .reset_index()
        .rename(columns={
            ycol: "event_year",
            ncol: "event_name",
            "AirTemp": "mean_air_temp",
            "TrackTemp": "mean_track_temp",
            "Humidity": "mean_humidity",
            "Rainfall": "rain_any",
            "WindSpeed": "mean_wind_speed",
            "WindDirection": "mean_wind_dir"
        })
    )
    
    wx_feat["rain_any"] = wx_feat.get("rain_any", 0).fillna(0)
    wx_feat["is_wet_flag"] = (wx_feat["rain_any"] > 0).astype(int)

    # Cyclic encoding for wind direction
    if "mean_wind_dir" in wx_feat.columns:
        rad = np.deg2rad(wx_feat["mean_wind_dir"])
        wx_feat["wind_sin"] = np.sin(rad)
        wx_feat["wind_cos"] = np.cos(rad)
    else:
        wx_feat["wind_sin"] = np.nan
        wx_feat["wind_cos"] = np.nan

    return wx_feat


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Standard feature columns used across training and inference
FEATURE_COLS = [
    "grid_pos", "quali_gap_s", "grid_quali_diff",
    "driver_last3_avg_finish", "team_last3_avg_finish",
    "pos_change", "race_total_overtakes",
    "driver_overtakes", "driver_times_overtaken", "driver_net_passes",
    "pit_loss_total_s",
    "mean_air_temp", "mean_track_temp", "mean_wind_speed", "mean_wind_dir", "wind_sin", "wind_cos",
    "is_wet_flag", "start_compound", "start_soft", "start_medium", "start_hard", "start_inter", "start_wet"
]

# Columns to zero-fill for missing values
ZERO_FILL_COLS = [
    "quali_gap_s", "pit_loss_total_s",
    "mean_air_temp", "mean_track_temp", "mean_wind_speed", "mean_wind_dir", "wind_sin", "wind_cos",
    "pos_change", "driver_overtakes", "driver_times_overtaken", "driver_net_passes"
]

# Metadata columns (not features)
META_COLS = {"race_id", "event_year", "event_name", "Driver", "DriverNumber", "TeamName"}
TARGET_COLS = {"finish_pos", "scored_points"}


def add_weather_columns(df: pd.DataFrame, wx_feat: Optional[pd.DataFrame], drop_humidity: bool = True) -> pd.DataFrame:
    """
    Merge weather features into main dataframe.
    
    Args:
        df: Main dataframe
        wx_feat: Weather features dataframe (or None)
        drop_humidity: Whether to drop mean_humidity column
        
    Returns:
        DataFrame with weather columns added
    """
    if wx_feat is not None:
        df = df.merge(wx_feat, on=["event_year", "event_name"], how="left")
        df["is_wet_flag"] = df.get("is_wet_flag", 0).fillna(0).astype(int)
        if drop_humidity and "mean_humidity" in df.columns:
            df = df.drop(columns=["mean_humidity"])
    else:
        for c in ["mean_air_temp", "mean_track_temp", "mean_wind_speed", "mean_wind_dir", "wind_sin", "wind_cos"]:
            df[c] = np.nan
        df["is_wet_flag"] = 0
    return df


def add_starting_compound(df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add starting tyre compound features from lap data.
    
    Args:
        df: Main dataframe
        laps: Laps dataframe with Compound column
        
    Returns:
        DataFrame with compound columns added
    """
    if laps is None or laps.empty:
        df["start_compound"] = np.nan
        for comp in ["soft", "medium", "hard", "inter", "wet"]:
            df[f"start_{comp}"] = np.nan
        return df

    # Ensure required columns exist
    for c in ["event_year", "event_name", "Driver", "LapNumber", "Compound"]:
        if c not in laps.columns:
            laps[c] = np.nan

    # Get first lap compound for each driver
    tyre_first = (
        laps
        .sort_values(["event_year", "event_name", "Driver", "LapNumber"])
        .dropna(subset=["Compound"])
        .groupby(["event_year", "event_name", "Driver"], as_index=False)
        .first()[["event_year", "event_name", "Driver", "Compound"]]
        .rename(columns={"Compound": "start_compound"})
    )

    df = df.merge(tyre_first, on=["event_year", "event_name", "Driver"], how="left")
    df["start_compound"] = df["start_compound"].astype(str).str.strip().str.title()

    # One-hot encoding for compounds
    for comp in ["Soft", "Medium", "Hard", "Inter", "Wet"]:
        df[f"start_{comp.lower()}"] = (df["start_compound"] == comp).astype("Int64")

    return df

