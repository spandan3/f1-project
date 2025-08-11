import pandas as pd, numpy as np
from pathlib import Path

# Project root (two levels up from backend/data/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_LAPS = BASE_DIR / "data" / "raw" / "laps.parquet"
RAW_RES  = BASE_DIR / "data" / "raw" / "results.parquet"
RAW_WX   = BASE_DIR / "data" / "raw" / "weather.parquet"
FE_DIR   = BASE_DIR / "data" / "fe"
FE_DIR.mkdir(parents=True, exist_ok=True)

def _first_col(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if default and default not in df.columns:
        df[default] = np.nan
        return default
    tmp = f"__missing_{candidates[0]}"
    df[tmp] = np.nan
    return tmp

def _to_seconds(x):
    if pd.isna(x): 
        return np.nan
    if isinstance(x, pd.Timedelta):
        return x.total_seconds()
    td = pd.to_timedelta(str(x), errors="coerce")
    if not pd.isna(td):
        return td.total_seconds()
    try:
        m, s = str(x).split(":")
        return int(m)*60 + float(s)
    except Exception:
        return np.nan

def _load_weather_features() -> pd.DataFrame | None:
    if not RAW_WX.exists():
        return None
    wx = pd.read_parquet(RAW_WX).copy()

    # Session type col
    st_col = "session_type" if "session_type" in wx.columns else ("SessionType" if "SessionType" in wx.columns else None)
    if st_col is None:
        return None
    # Keep race session weather only
    wx_race = wx[wx[st_col] == "R"].copy()

    # Column name mapping
    year_col = _first_col(wx_race, ["event_year", "EventYear"], default="event_year")
    name_col = _first_col(wx_race, ["event_name", "EventName"], default="event_name")

    # Aggregate per event
    agg = {}
    if "AirTemp" in wx_race.columns:   agg["AirTemp"] = "mean"
    if "TrackTemp" in wx_race.columns: agg["TrackTemp"] = "mean"
    if "Humidity" in wx_race.columns:  agg["Humidity"] = "mean"
    if "Rainfall" in wx_race.columns:  agg["Rainfall"] = "max"

    if not agg:
        return None

    wx_feat = wx_race.groupby([year_col, name_col]).agg(agg).reset_index()
    wx_feat = wx_feat.rename(columns={
        year_col: "event_year",
        name_col: "event_name",
        "AirTemp": "mean_air_temp",
        "TrackTemp": "mean_track_temp",
        "Humidity": "mean_humidity",
        "Rainfall": "rain_any"
    })
    wx_feat["rain_any"] = wx_feat.get("rain_any", 0).fillna(0)
    wx_feat["is_wet_flag"] = (wx_feat["rain_any"] > 0).astype(int)

    return wx_feat

def make_pre_race_table() -> pd.DataFrame:
    res = pd.read_parquet(RAW_RES)

    # Session type column can vary
    st_col = "session_type" if "session_type" in res.columns else ("SessionType" if "SessionType" in res.columns else None)
    if st_col is None:
        raise ValueError("results.parquet has no session type column")

    # Identify flexible column names
    drv_col   = _first_col(res, ["Driver", "Abbreviation", "DriverId"], default="Driver")
    team_col  = _first_col(res, ["TeamName", "Team"], default="TeamName")
    grid_col  = _first_col(res, ["GridPosition"], default="GridPosition")
    pos_col   = _first_col(res, ["Position"], default="Position")
    stat_col  = _first_col(res, ["Status"], default="Status")
    year_col  = _first_col(res, ["event_year", "EventYear"], default="event_year")
    name_col  = _first_col(res, ["event_name", "EventName"], default="event_name")
    dnum_col  = _first_col(res, ["DriverNumber"], default="DriverNumber")

    # --- Qualifying slice ---
    quali = res[res[st_col] == "Q"].copy()
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
        grid_col: "GridPosition",
    })
    quali["q1_s"] = quali["Q1"].map(_to_seconds)
    quali["q2_s"] = quali["Q2"].map(_to_seconds)
    quali["q3_s"] = quali["Q3"].map(_to_seconds)
    quali["best_q"] = quali[["q1_s","q2_s","q3_s"]].min(axis=1, skipna=True)
    quali["quali_gap_s"] = quali["best_q"] - quali.groupby(["event_year","event_name"])["best_q"].transform("min")
    quali["grid_pos"] = quali["QualiPos"]
    quali.loc[quali["grid_pos"].isna() & quali["GridPosition"].notna(), "grid_pos"] = quali["GridPosition"]
    quali_feat = quali[["DriverNumber","event_year","event_name","grid_pos","quali_gap_s"]].copy()

    # --- Race slice ---
    race = res[res[st_col] == "R"].copy().rename(columns={
        drv_col: "Driver",
        team_col: "TeamName",
        year_col: "event_year",
        name_col: "event_name",
        dnum_col: "DriverNumber",
        pos_col: "finish_pos",
        stat_col: "Status",
        grid_col: "GridPosition"
    })

    # Merge quali → race
    df = pd.merge(
        race[["DriverNumber","Driver","TeamName","event_year","event_name","finish_pos","Status","GridPosition"]],
        quali_feat,
        on=["DriverNumber","event_year","event_name"],
        how="left"
    )
    df.loc[df["grid_pos"].isna() & df["GridPosition"].notna(), "grid_pos"] = df["GridPosition"]
    df = df.drop(columns=["GridPosition"])

    # Rolling form features
    df = df.sort_values(["Driver","event_year","event_name"], kind="mergesort")
    df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")

    grp_d = df.groupby("Driver", group_keys=False)
    df["driver_last3_avg_finish"] = grp_d["finish_pos"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    status_str = df["Status"].astype(str).fillna("")
    df["dnf_flag"] = ((df["finish_pos"].isna()) | (status_str != "Finished")).astype(int)
    df["driver_last3_dnfs"] = (
        df.groupby("Driver")["dnf_flag"]
          .apply(lambda s: s.shift().rolling(3, min_periods=1).sum())
          .reset_index(level=0, drop=True)
    )
    grp_t = df.groupby("TeamName", group_keys=False)
    df["team_last3_avg_finish"] = grp_t["finish_pos"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())

    # --- Weather merge ---
    wx_feat = _load_weather_features()
    if wx_feat is not None:
        df = df.merge(wx_feat, on=["event_year","event_name"], how="left")
        df["is_wet_flag"] = df["is_wet_flag"].fillna(df["is_wet_flag"])
    else:
        df["mean_air_temp"] = np.nan
        df["mean_track_temp"] = np.nan
        df["mean_humidity"] = np.nan
        df["is_wet_flag"] = 0

    # Track descriptors (still placeholder)
    df["track_overtake_idx"] = 0.5
    df["pit_loss_s"] = 22.0

    # Labels / race id
    df["relevance"] = (df["finish_pos"]==1)*3 + (df["finish_pos"]==2)*2 + (df["finish_pos"]==3)*1
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)

    # Final dataset
    features = [
        "grid_pos","quali_gap_s","driver_last3_avg_finish","team_last3_avg_finish",
        "track_overtake_idx","pit_loss_s","is_wet_flag",
        "mean_air_temp","mean_track_temp","mean_humidity"
    ]
    keep = ["race_id","Driver","DriverNumber","TeamName","finish_pos","relevance"] + features
    out = df[keep].copy()
    out = out.dropna(subset=["grid_pos"], how="any")
    out["quali_gap_s"] = out["quali_gap_s"].fillna(0.0)

    # Save
    out_path = FE_DIR / "standings_train.parquet"
    out.to_parquet(out_path, index=False)
    print(f"✅ Saved features to {out_path} (rows={len(out)}, cols={out.shape[1]})")
    return out

if __name__=="__main__":
    make_pre_race_table()
