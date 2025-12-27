# backend/ml/build_inference_rows.py
"""
Build pre-race inference feature table using ONLY pre-race info.
Loads sessions directly from FastF1's on-disk cache.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from .common import (
    BASE_DIR,
    CACHE_DIR,
    FE_DIR,
    to_seconds,
    PRE_RACE_FEATURES,
)

# Inference output directory
OUT_DIR = FE_DIR / "inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(CACHE_DIR))


def _normalize_results(df: pd.DataFrame, year: int, event_name: str, session_type: str) -> pd.DataFrame:
    """
    Build a fresh, de-duplicated results frame with a consistent schema.
    
    Priority rules:
        Driver: Driver > Abbreviation > DriverId
        Team: TeamName > Team
    """
    src = df.copy()

    def pick(*cands):
        for c in cands:
            if c in src.columns:
                return src[c]
        return pd.Series([np.nan] * len(src))

    out = pd.DataFrame({
        "Driver": pick("Driver", "Abbreviation", "DriverId"),
        "TeamName": pick("TeamName", "Team"),
        "DriverNumber": pick("DriverNumber"),
        "Position": pick("Position"),
        "GridPosition": pick("GridPosition"),
        "Q1": pick("Q1"),
        "Q2": pick("Q2"),
        "Q3": pick("Q3"),
    })

    out["event_year"] = year
    out["event_name"] = event_name
    out["session_type"] = session_type

    # Ensure dtypes
    out["Driver"] = out["Driver"].astype(str)
    out["TeamName"] = out["TeamName"].astype(str)
    for c in ["DriverNumber", "Position", "GridPosition"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _event_list_upto(year: int, event_name: str, n_back: int = 6) -> list[tuple[int, str]]:
    """
    Return list of up to n_back RACE events BEFORE the given event (chronological backward).
    Walks into previous season if needed.
    """
    lst: list[tuple[int, str]] = []

    def season_events(y_):
        cal = fastf1.get_event_schedule(y_)
        return cal["EventName"].tolist()

    # Current season: take events before the target
    current = season_events(year)
    if event_name in current:
        idx = current.index(event_name)
        for i in range(idx - 1, -1, -1):
            lst.append((year, current[i]))
            if len(lst) >= n_back:
                return lst

    # Walk into previous seasons if needed
    py = year - 1
    while len(lst) < n_back and py >= 2018:
        prev = season_events(py)
        for en in reversed(prev):
            lst.append((py, en))
            if len(lst) >= n_back:
                break
        py -= 1
        
    return lst[:n_back]


def _rolling_form_lastN_races(year: int, event_name: str, n_back: int = 6) -> pd.DataFrame:
    """
    Compute driver/team rolling form using the last N finished races.
    Only loads those races from cache (fast if cached).
    """
    events = _event_list_upto(year, event_name, n_back=n_back)
    rows = []
    
    for (yy, en) in events:
        try:
            r = fastf1.get_session(yy, en, "R")
            r.load()
            res = _normalize_results(r.results, yy, r.event["EventName"], "R")
            rows.append(res[["Driver", "TeamName", "DriverNumber", "event_year", "event_name", "Position"]])
        except Exception:
            continue
            
    if not rows:
        return pd.DataFrame(columns=["Driver", "TeamName", "DriverNumber", 
                                      "driver_last3_avg_finish", "team_last3_avg_finish"])

    df = pd.concat(rows, ignore_index=True)
    df["finish_pos"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.sort_values(["Driver", "event_year", "event_name"])

    last3 = lambda s: s.shift().rolling(3, min_periods=1).mean()
    df["driver_last3_avg_finish"] = df.groupby("Driver", group_keys=False)["finish_pos"].transform(last3)
    df["team_last3_avg_finish"] = df.groupby("TeamName", group_keys=False)["finish_pos"].transform(last3)

    # Keep only the LATEST value per DriverNumber
    latest = (
        df.sort_values(["event_year", "event_name"])
        .groupby(["Driver", "TeamName", "DriverNumber"], as_index=False)
        .tail(1)[["Driver", "TeamName", "DriverNumber", "driver_last3_avg_finish", "team_last3_avg_finish"]]
    )
    return latest


def _practice_features(year: int, event_name: str) -> pd.DataFrame:
    """
    Extract practice session features (FP2, FP3).
    
    Returns:
        DataFrame with best lap times from practice sessions
    """
    frames = []
    for kind in ["FP2", "FP3"]:
        try:
            s = fastf1.get_session(year, event_name, kind)
            s.load()
            laps = s.laps
            if "Driver" not in laps.columns or "LapTime" not in laps.columns:
                continue
            best = (
                laps.groupby("Driver")["LapTime"]
                .min()
                .reset_index()
                .rename(columns={"LapTime": f"best_{kind.lower()}_s"})
            )
            best[f"best_{kind.lower()}_s"] = best[f"best_{kind.lower()}_s"].map(to_seconds)
            frames.append(best)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["Driver", "best_fp2_s", "best_fp3_s", "mean_best_lap_practice_s"])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Driver", how="outer")

    for c in ["best_fp2_s", "best_fp3_s"]:
        if c not in out.columns:
            out[c] = np.nan

    out["mean_best_lap_practice_s"] = out[["best_fp2_s", "best_fp3_s"]].mean(axis=1, skipna=True)
    out["Driver"] = out["Driver"].astype(str)
    
    return out[["Driver", "best_fp2_s", "best_fp3_s", "mean_best_lap_practice_s"]]


def _load_forecast_csv(path: str | None) -> pd.DataFrame | None:
    """Load optional weather forecast CSV."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Forecast file not found: {p}")
    df = pd.read_csv(p)
    need = {"event_year", "event_name"}
    if not need.issubset(df.columns):
        raise ValueError(f"Forecast CSV must contain: {need}")
    return df


def build_inference_from_cache(
    event_year: int,
    event_name: str,
    n_hist_races: int = 6,
    use_practice: bool = False,
    forecast_csv: str | None = None
) -> Path:
    """
    Build pre-race inference features from cached FastF1 data.
    
    Args:
        event_year: Year of the target race
        event_name: Name of the target race event
        n_hist_races: Number of historical races for rolling form
        use_practice: Whether to include FP2/FP3 features
        forecast_csv: Optional path to weather forecast CSV
        
    Returns:
        Path to saved inference parquet file
    """
    # === Qualifying (target event) ===
    q = fastf1.get_session(event_year, event_name, "Q")
    q.load()
    q_res = _normalize_results(q.results, event_year, q.event["EventName"], "Q")

    # Qualifying features
    quali = q_res.copy()
    for qcol in ["Q1", "Q2", "Q3"]:
        if qcol not in quali.columns:
            quali[qcol] = np.nan
            
    quali["q1_s"] = quali["Q1"].map(to_seconds)
    quali["q2_s"] = quali["Q2"].map(to_seconds)
    quali["q3_s"] = quali["Q3"].map(to_seconds)
    quali["best_q"] = quali[["q1_s", "q2_s", "q3_s"]].min(axis=1, skipna=True)
    pole = quali.groupby(["event_year", "event_name"])["best_q"].transform("min")
    quali["quali_gap_s"] = (quali["best_q"] - pole).fillna(0.0)
    
    # Grid position pre-race = Quali Position (penalties unknown)
    quali["grid_pos"] = pd.to_numeric(quali["Position"], errors="coerce")
    quali.loc[quali["grid_pos"].isna() & quali["GridPosition"].notna(), "grid_pos"] = \
        pd.to_numeric(quali["GridPosition"], errors="coerce")
    quali["grid_quali_diff"] = 0.0  # Unknown pre-race

    # === Rolling form from LAST N races ===
    form = _rolling_form_lastN_races(event_year, event_name, n_back=n_hist_races)
    df = quali.merge(form, on=["Driver", "TeamName", "DriverNumber"], how="left")

    # Fill form gaps
    for c in ["driver_last3_avg_finish", "team_last3_avg_finish"]:
        if c not in df.columns:
            df[c] = np.nan
    df["driver_last3_avg_finish"] = df.groupby("Driver")["driver_last3_avg_finish"].transform(lambda s: s.fillna(s.mean()))
    df["team_last3_avg_finish"] = df.groupby("TeamName")["team_last3_avg_finish"].transform(lambda s: s.fillna(s.mean()))
    df["driver_last3_avg_finish"] = df["driver_last3_avg_finish"].fillna(df["driver_last3_avg_finish"].mean())
    df["team_last3_avg_finish"] = df["team_last3_avg_finish"].fillna(df["team_last3_avg_finish"].mean())

    # === Optional practice features ===
    if use_practice:
        pfx = _practice_features(event_year, event_name)
        if not pfx.empty:
            df = df.merge(pfx, on="Driver", how="left")
        else:
            for c in ["best_fp2_s", "best_fp3_s", "mean_best_lap_practice_s"]:
                df[c] = np.nan
    else:
        for c in ["best_fp2_s", "best_fp3_s", "mean_best_lap_practice_s"]:
            df[c] = np.nan

    # === Weather forecast (optional) ===
    fcast = _load_forecast_csv(forecast_csv)
    if fcast is not None:
        fcast = fcast.copy()
        df = df.merge(fcast, on=["event_year", "event_name"], how="left")
        for c in ["mean_air_temp", "mean_track_temp", "mean_wind_speed", "mean_wind_dir", "is_wet_flag"]:
            if c not in df.columns:
                df[c] = 0.0
        if "mean_wind_dir" in df.columns:
            rad = np.deg2rad(df["mean_wind_dir"].fillna(0.0))
            df["wind_sin"] = np.sin(rad)
            df["wind_cos"] = np.cos(rad)
        else:
            df["wind_sin"] = 0.0
            df["wind_cos"] = 0.0
    else:
        # Neutral defaults when no forecast
        df["mean_air_temp"] = 0.0
        df["mean_track_temp"] = 0.0
        df["mean_wind_speed"] = 0.0
        df["mean_wind_dir"] = 0.0
        df["wind_sin"] = 0.0
        df["wind_cos"] = 0.0
        df["is_wet_flag"] = 0

    # === Unknown pre-race features → neutral ===
    df["start_compound"] = pd.Series(["Unknown"] * len(df), dtype="category")
    for comp in ["soft", "medium", "hard", "inter", "wet"]:
        df[f"start_{comp}"] = 0

    # === Final columns (align with model) ===
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)
    
    # Use only PRE_RACE_FEATURES (no leaky post-race features)
    feature_cols = PRE_RACE_FEATURES + ["best_fp2_s", "best_fp3_s", "mean_best_lap_practice_s"]
    keep = ["race_id", "event_year", "event_name", "Driver", "DriverNumber", "TeamName"] + \
           [c for c in feature_cols if c in df.columns]
    out = df[keep].copy()

    # Sanity checks
    out["grid_pos"] = pd.to_numeric(out["grid_pos"], errors="coerce")
    out = out.dropna(subset=["grid_pos"])

    # Save
    fe_path = OUT_DIR / f"infer_{event_year}_{event_name.replace(' ', '_')}.parquet"
    out.to_parquet(fe_path, index=False)
    print(f"✅ Wrote inference features → {fe_path} (rows={len(out)}, cols={out.shape[1]})")
    
    return fe_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build pre-race inference rows from FastF1 cache.")
    ap.add_argument("--year", type=int, required=True, help="Event year, e.g. 2025")
    ap.add_argument("--event", type=str, required=True, help="Exact EventName, e.g. 'Australian Grand Prix'")
    ap.add_argument("--n-hist", type=int, default=6, help="Past races for rolling form (default: 6)")
    ap.add_argument("--practice", action="store_true", help="Include FP2/FP3 practice features")
    ap.add_argument("--forecast-csv", type=str, default=None, help="Optional weather forecast CSV")
    args = ap.parse_args()
    
    build_inference_from_cache(args.year, args.event, args.n_hist, args.practice, args.forecast_csv)
