import pandas as pd, numpy as np
from pathlib import Path

# Base project directory (two levels up from backend/data/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_LAPS = BASE_DIR / "data" / "raw" / "laps.parquet"
RAW_RES  = BASE_DIR / "data" / "raw" / "results.parquet"
RAW_WX   = BASE_DIR / "data" / "raw" / "weather.parquet"
FE_DIR   = BASE_DIR / "data" / "fe"
FE_DIR.mkdir(parents=True, exist_ok=True)

def _to_seconds(x):
    if pd.isna(x): return np.nan
    if isinstance(x, pd.Timedelta): return x.total_seconds()
    td = pd.to_timedelta(str(x), errors="coerce")
    if not pd.isna(td): return td.total_seconds()
    try:
        m, s = str(x).split(":")
        return int(m)*60 + float(s)
    except Exception:
        return np.nan

def _load_weather_features() -> pd.DataFrame | None:
    if not RAW_WX.exists():
        return None
    wx = pd.read_parquet(RAW_WX).copy()

    # Detect session type column name
    st_col = "session_type" if "session_type" in wx.columns else ("SessionType" if "SessionType" in wx.columns else None)
    if st_col is None:
        return None

    # Keep RACE weather only (pre-race feature should come from race-day conditions; for live use, you’ll feed forecast)
    wx_r = wx[wx[st_col] == "R"].copy()

    # Normalise expected columns if present
    cols = wx_r.columns
    air = "AirTemp" if "AirTemp" in cols else None
    track = "TrackTemp" if "TrackTemp" in cols else None
    rain = "Rainfall" if "Rainfall" in cols else None
    hum = "Humidity" if "Humidity" in cols else None
    wind = "WindSpeed" if "WindSpeed" in cols else None

    agg_dict = {}
    if air:   agg_dict["mean_air_temp"] = (air, "mean")
    if track: agg_dict["mean_track_temp"] = (track, "mean")
    if hum:   agg_dict["mean_humidity"] = (hum, "mean")
    if wind:  agg_dict["mean_wind_speed"] = (wind, "mean")
    if rain:
        # any rainfall during race session → wet flag
        agg_dict["rain_any"] = (rain, lambda x: int(pd.Series(x).astype(bool).any()))

    if not agg_dict:
        return None

    wx_feat = wx_r.groupby(["event_year", "event_name"]).agg(**agg_dict).reset_index()

    # If rain_any missing (no Rainfall col), create it as 0
    if "rain_any" not in wx_feat.columns:
        wx_feat["rain_any"] = 0

    return wx_feat

def make_pre_race_table() -> pd.DataFrame:
    # Load raw parquet
    res = pd.read_parquet(RAW_RES)

    # --- Qualifying features ---
    quali_mask = (res.get("session_type") == "Q") if "session_type" in res.columns else res.get("SessionType") == "Q"
    quali = res[quali_mask].copy()

    race_mask = (res.get("session_type") == "R") if "session_type" in res.columns else res.get("SessionType") == "R"
    race = res[race_mask].copy()

    # Ensure columns exist
    for c in ["DriverNumber","Driver","TeamName","event_name","event_year","Position","Q1","Q2","Q3","GridPosition","Status"]:
        if c not in quali.columns: quali[c] = np.nan
        if c not in race.columns:  race[c]  = np.nan

    quali["q1_s"] = quali["Q1"].map(_to_seconds)
    quali["q2_s"] = quali["Q2"].map(_to_seconds)
    quali["q3_s"] = quali["Q3"].map(_to_seconds)
    quali["best_q"] = quali[["q1_s","q2_s","q3_s"]].min(axis=1, skipna=True)
    quali["quali_gap_s"] = quali["best_q"] - quali.groupby(["event_year","event_name"])["best_q"].transform("min")

    # Grid from quali Position with fallback to GridPosition
    quali["grid_pos"] = quali["Position"]
    quali.loc[quali["grid_pos"].isna() & quali["GridPosition"].notna(), "grid_pos"] = quali["GridPosition"]

    quali_feat = quali[["DriverNumber","event_year","event_name","grid_pos","quali_gap_s"]].copy()

    # --- Race labels ---
    race = race.rename(columns={"Position": "finish_pos"})
    df = pd.merge(
        race[["DriverNumber","Driver","TeamName","event_name","event_year","finish_pos","Status"]],
        quali_feat,
        on=["DriverNumber","event_year","event_name"],
        how="left"
    )
    # Fallback grid from race if still missing
    if "GridPosition" in race.columns:
        df.loc[df["grid_pos"].isna() & race["GridPosition"].notna(), "grid_pos"] = race["GridPosition"].values

    # --- Rolling forms ---
    df = df.sort_values(["Driver","event_year","event_name"], kind="mergesort")
    df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")

    grp_d = df.groupby("Driver", group_keys=False)
    df["driver_last3_avg_finish"] = grp_d["finish_pos"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())

    def _dnf_series(g):
        return ((g["finish_pos"].isna()) | (g.get("Status", pd.Series(index=g.index)).astype(str) != "Finished")).astype(int)
    df["driver_last3_dnfs"] = grp_d.apply(lambda g: _dnf_series(g).shift().rolling(3, min_periods=1).sum()).reset_index(level=0, drop=True)

    grp_t = df.groupby("TeamName", group_keys=False)
    df["team_last3_avg_finish"] = grp_t["finish_pos"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())

    # --- Weather join (from weather.parquet) ---
    wx_feat = _load_weather_features()
    if wx_feat is not None:
        df = df.merge(wx_feat, on=["event_year","event_name"], how="left")
        # real wet flag if we have rainfall; else 0
        df["is_wet_flag"] = df.get("rain_any", 0).fillna(0).astype(int)
        # Optional: keep temps as features
        df["mean_air_temp"] = df.get("mean_air_temp", np.nan)
        df["mean_track_temp"] = df.get("mean_track_temp", np.nan)
    else:
        # Fallback if no weather file/cols
        df["is_wet_flag"] = 0
        df["mean_air_temp"] = np.nan
        df["mean_track_temp"] = np.nan

    # --- Track placeholders (replace later with real per-circuit table) ---
    df["track_overtake_idx"] = 0.5
    df["pit_loss_s"] = 22.0

    # Labels / groups
    df["relevance"] = (df["finish_pos"]==1)*3 + (df["finish_pos"]==2)*2 + (df["finish_pos"]==3)*1
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)

    # Features to keep
    features = [
        "grid_pos","quali_gap_s","driver_last3_avg_finish","team_last3_avg_finish",
        "track_overtake_idx","pit_loss_s","is_wet_flag","mean_air_temp","mean_track_temp"
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

if __name__ == "__main__":
    make_pre_race_table()