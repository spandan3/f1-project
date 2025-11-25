# backend/data/build_dataset.py
import pandas as pd, numpy as np
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_LAPS = BASE_DIR / "data" / "raw" / "laps.parquet"
RAW_RES  = BASE_DIR / "data" / "raw" / "results.parquet"
RAW_WX   = BASE_DIR / "data" / "raw" / "weather.parquet"
PROC_DIR = BASE_DIR / "data" / "processed"
FE_DIR   = BASE_DIR / "data" / "fe"
PROC_DIR.mkdir(parents=True, exist_ok=True)
FE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Small helpers ----------
def _first_col(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str:
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

# ---------- Weather aggregation (incl wind + wet flag) ----------
def _load_weather_features() -> pd.DataFrame | None:
    if not RAW_WX.exists():
        return None
    wx = pd.read_parquet(RAW_WX).copy()

    st_col = _first_col(wx, ["session_type","SessionType"], None)
    if st_col is None:
        return None

    wx_race = wx[wx[st_col].astype(str) == "R"].copy()
    year_col = _first_col(wx_race, ["event_year","EventYear"], "event_year")
    name_col = _first_col(wx_race, ["event_name","EventName"], "event_name")

    agg = {}
    if "AirTemp" in wx_race:       agg["AirTemp"] = "mean"
    if "TrackTemp" in wx_race:     agg["TrackTemp"] = "mean"
    if "Humidity" in wx_race:      agg["Humidity"] = "mean"
    if "Rainfall" in wx_race:      agg["Rainfall"] = "max"
    if "WindSpeed" in wx_race:     agg["WindSpeed"] = "mean"
    if "WindDirection" in wx_race: agg["WindDirection"] = "mean"
    if not agg:
        return None

    wx_feat = (wx_race
               .groupby([year_col, name_col])
               .agg(agg)
               .reset_index()
               .rename(columns={
                   year_col: "event_year",
                   name_col: "event_name",
                   "AirTemp": "mean_air_temp",
                   "TrackTemp": "mean_track_temp",
                   "Humidity": "mean_humidity",
                   "Rainfall": "rain_any",
                   "WindSpeed": "mean_wind_speed",
                   "WindDirection": "mean_wind_dir"
               }))
    wx_feat["rain_any"] = wx_feat.get("rain_any", 0).fillna(0)
    wx_feat["is_wet_flag"] = (wx_feat["rain_any"] > 0).astype(int)

    # cyclic encoding for direction
    if "mean_wind_dir" in wx_feat.columns:
        rad = np.deg2rad(wx_feat["mean_wind_dir"])
        wx_feat["wind_sin"] = np.sin(rad)
        wx_feat["wind_cos"] = np.cos(rad)
    else:
        wx_feat["mean_wind_dir"] = np.nan
        wx_feat["wind_sin"] = np.nan
        wx_feat["wind_cos"] = np.nan

    return wx_feat

# ---------- Pit loss per stop then sum per driver ----------
def _compute_pit_losses_sum(laps: pd.DataFrame, n_baseline: int = 3) -> pd.DataFrame:
    """
    pit_loss = (in_lap_time + out_lap_time) - 2 * baseline_lap_time
    baseline_lap_time = median of previous n clean laps (no pit in/out).
    Returns total pit loss per driver per event.
    """
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["event_year","event_name","DriverNumber","pit_loss_total_s"])

    year_col = _first_col(laps, ["event_year","EventYear"], "event_year")
    name_col = _first_col(laps, ["event_name","EventName"], "event_name")
    dnum_col = _first_col(laps, ["DriverNumber"], "DriverNumber")
    drv_col  = _first_col(laps, ["Driver","Abbreviation"], "Driver")

    # Ensure needed columns exist
    for c in ["LapNumber","LapTime","PitInTime","PitOutTime"]:
        if c not in laps.columns:
            laps[c] = np.nan

    df = (laps[[year_col,name_col,dnum_col,drv_col,"LapNumber","LapTime","PitInTime","PitOutTime"]]
           .copy()
           .rename(columns={year_col:"event_year", name_col:"event_name",
                            dnum_col:"DriverNumber", drv_col:"Driver"})
           .sort_values(["Driver","LapNumber"]))

    def _sec(x):
        if pd.isna(x): return np.nan
        if isinstance(x, pd.Timedelta): return float(x.total_seconds())
        td = pd.to_timedelta(x, errors="coerce")
        if not pd.isna(td): return float(td.total_seconds())
        return float(x) if isinstance(x,(int,float)) else np.nan

    rows = []
    for (ey,en,dr,dnum), g in df.groupby(["event_year","event_name","Driver","DriverNumber"], sort=False):
        g = g.sort_values("LapNumber")
        out_idx = g.index[g["PitOutTime"].notna()].tolist()
        for oi in out_idx:
            out_row = g.loc[oi]
            out_lap = int(out_row["LapNumber"])
            in_rows = g[g["LapNumber"] == out_lap - 1]
            if in_rows.empty:
                continue
            in_row = in_rows.iloc[0]

            prev = g[(g["LapNumber"] < out_lap) &
                     (g["PitInTime"].isna()) &
                     (g["PitOutTime"].isna()) &
                     (g["LapTime"].notna())].tail(n_baseline)
            if len(prev) == 0:
                baseline = _sec(g[(g["PitInTime"].isna()) &
                                  (g["PitOutTime"].isna()) &
                                  (g["LapTime"].notna())]["LapTime"].median())
            else:
                baseline = _sec(prev["LapTime"].median())

            in_lt  = _sec(in_row["LapTime"])
            out_lt = _sec(out_row["LapTime"])
            loss = np.nan if any(np.isnan([baseline,in_lt,out_lt])) else (in_lt + out_lt - 2.0*baseline)

            rows.append({"event_year":ey,"event_name":en,"DriverNumber":dnum,"pit_loss_s":loss})

    per_stop = pd.DataFrame(rows)
    if per_stop.empty:
        return pd.DataFrame(columns=["event_year","event_name","DriverNumber","pit_loss_total_s"])

    total = (per_stop.groupby(["event_year","event_name","DriverNumber"])["pit_loss_s"]
                    .sum()
                    .reset_index()
                    .rename(columns={"pit_loss_s":"pit_loss_total_s"}))
    return total

# ---------- Main: load, engineer, save ----------
def make_pre_race_table() -> pd.DataFrame:
    # Read raw
    res  = pd.read_parquet(RAW_RES)
    laps = pd.read_parquet(RAW_LAPS) if RAW_LAPS.exists() else pd.DataFrame()

    # Flexible column names
    st_col  = _first_col(res, ["session_type","SessionType"], None)
    if st_col is None:
        raise ValueError("results.parquet has no session type column")

    drv_col  = _first_col(res, ["Driver","Abbreviation","DriverId"], "Driver")
    team_col = _first_col(res, ["TeamName","Team"], "TeamName")
    grid_col = _first_col(res, ["GridPosition"], "GridPosition")
    pos_col  = _first_col(res, ["Position"], "Position")
    stat_col = _first_col(res, ["Status"], "Status")
    year_col = _first_col(res, ["event_year","EventYear"], "event_year")
    name_col = _first_col(res, ["event_name","EventName"], "event_name")
    dnum_col = _first_col(res, ["DriverNumber"], "DriverNumber")

    # ---- Qualifying slice ----
    quali = res[res[st_col].astype(str) == "Q"].copy()
    for q in ["Q1","Q2","Q3"]:
        if q not in quali.columns:
            quali[q] = np.nan

    quali = quali.rename(columns={
        drv_col:"Driver", team_col:"TeamName",
        year_col:"event_year", name_col:"event_name",
        dnum_col:"DriverNumber", pos_col:"QualiPos", grid_col:"GridPosition"
    })
    quali["q1_s"] = quali["Q1"].map(_to_seconds)
    quali["q2_s"] = quali["Q2"].map(_to_seconds)
    quali["q3_s"] = quali["Q3"].map(_to_seconds)
    quali["best_q"] = quali[["q1_s","q2_s","q3_s"]].min(axis=1, skipna=True)
    quali["quali_gap_s"] = quali["best_q"] - quali.groupby(["event_year","event_name"])["best_q"].transform("min")
    quali["quali_pos_final"] = quali["QualiPos"].where(quali["QualiPos"].notna(), quali["GridPosition"])

    quali_feat = quali[["DriverNumber","event_year","event_name","quali_pos_final","quali_gap_s"]].copy()

    # ---- Race slice ----
    race = res[res[st_col].astype(str) == "R"].copy().rename(columns={
        drv_col:"Driver", team_col:"TeamName",
        year_col:"event_year", name_col:"event_name",
        dnum_col:"DriverNumber", pos_col:"finish_pos",
        stat_col:"Status", grid_col:"GridPosition"
    })

    # Merge quali → race
    df = pd.merge(
        race[["DriverNumber","Driver","TeamName","event_year","event_name","finish_pos","Status","GridPosition"]],
        quali_feat,
        on=["DriverNumber","event_year","event_name"],
        how="left"
    )

    # Final grid and grid vs quali diff
    df["grid_pos"] = df["GridPosition"].where(df["GridPosition"].notna(), df["quali_pos_final"])
    df["grid_quali_diff"] = df["grid_pos"] - df["quali_pos_final"]
    df = df.drop(columns=["GridPosition"])

    # Sort and convert for rolling features
    # make sure rows are in chronological order within each driver/team
    df = df.sort_values(["Driver","event_year","event_name"], kind="mergesort").reset_index(drop=True)

    last3 = lambda s: s.shift().rolling(3, min_periods=1).mean()

    df["driver_last3_avg_finish"] = (
        df.groupby("Driver", group_keys=False)["finish_pos"].transform(last3)
    )
    df["team_last3_avg_finish"] = (
        df.groupby("TeamName", group_keys=False)["finish_pos"].transform(last3)
    )

    # Fill remaining NaNs (early season/rookies)
    for c in ["driver_last3_avg_finish","team_last3_avg_finish"]:
        df[c] = df.groupby("Driver" if c.startswith("driver") else "TeamName")[c].transform(
            lambda s: s.fillna(s.mean())
        )
        df[c] = df[c].fillna(df[c].mean())

    # Position change and total overtakes per race
    df["pos_change"] = df["grid_pos"] - df["finish_pos"]
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)
    overtakes = (df.assign(_gain=df["pos_change"].clip(lower=0))
                   .groupby("race_id")["_gain"].sum()
                   .rename("total_overtakes")
                   .reset_index())
    df = df.merge(overtakes, on="race_id", how="left")

    if "total_overtakes" in df.columns:
        df = df.rename(columns={"total_overtakes": "race_total_overtakes"})

# 2) Add per-driver overtakes from lap-by-lap position deltas
    if RAW_LAPS.exists():
        laps = pd.read_parquet(RAW_LAPS)

    # ensure required cols exist
        for c in ["event_year","event_name","Driver","LapNumber","Position"]:
            if c not in laps.columns:
                # can't compute without these; fall back to NaN columns
                df["driver_overtakes"] = np.nan
                df["driver_times_overtaken"] = np.nan
                df["driver_net_passes"] = np.nan
                break
            else:
                laps = laps.sort_values(["event_year","event_name","Driver","LapNumber"])
                # negative diff => moved up (overtook someone)
                pos_diff = laps.groupby(["event_year","event_name","Driver"])["Position"].diff()

                drv_ov = (pos_diff < 0).groupby(
                    [laps["event_year"], laps["event_name"], laps["Driver"]]
                ).sum().reset_index(name="driver_overtakes")

                drv_be = (pos_diff > 0).groupby(
                    [laps["event_year"], laps["event_name"], laps["Driver"]]
                ).sum().reset_index(name="driver_times_overtaken")

                drv = drv_ov.merge(drv_be, on=["event_year","event_name","Driver"], how="outer").fillna(0)
                drv["driver_net_passes"] = drv["driver_overtakes"] - drv["driver_times_overtaken"]

                dup_cols = [c for c in ["driver_overtakes", "driver_times_overtaken", "driver_net_passes"] if c in df.columns]
                if dup_cols:
                    df = df.drop(columns=dup_cols)

                df = df.merge(drv, on=["event_year","event_name","Driver"], how="left")
        else:
            df["driver_overtakes"] = np.nan
            df["driver_times_overtaken"] = np.nan
            df["driver_net_passes"] = np.nan

    # ---- Weather merge (incl wind + wet); then drop mean_humidity as requested ----
    wx_feat = _load_weather_features()
    if wx_feat is not None:
        df = df.merge(wx_feat, on=["event_year","event_name"], how="left")
        df["is_wet_flag"] = df["is_wet_flag"].fillna(0).astype(int)
        if "mean_humidity" in df.columns:
            df = df.drop(columns=["mean_humidity"])
    else:
        for c in ["mean_air_temp","mean_track_temp","mean_wind_speed","mean_wind_dir","wind_sin","wind_cos"]:
            df[c] = np.nan
        df["is_wet_flag"] = 0

    # ---- Actual pit loss from laps (sum per driver) ----
    if RAW_LAPS.exists():
        pit_tot = _compute_pit_losses_sum(pd.read_parquet(RAW_LAPS), n_baseline=3)
        df = df.merge(pit_tot, on=["event_year","event_name","DriverNumber"], how="left")
    else:
        df["pit_loss_total_s"] = np.nan
    

    # --- STARTING TYRE COMPOUND ---
    if RAW_LAPS.exists():
        laps_all = pd.read_parquet(RAW_LAPS)

        for c in ["event_year","event_name","Driver","LapNumber","Compound"]:
            if c not in laps_all.columns:
                laps_all[c] = np.nan

        # pick the first lap with a known compound for each driver in the race (usually Lap 1)
        tyre_first = (laps_all.sort_values(["event_year","event_name","Driver","LapNumber"])
                            .dropna(subset=["Compound"])
                            .groupby(["event_year","event_name","Driver"], as_index=False)
                            .first()[["event_year","event_name","Driver","Compound"]]
                            .rename(columns={"Compound":"start_compound"}))

        df = df.merge(tyre_first, on=["event_year","event_name","Driver"], how="left")

        # normalize to title case (Soft/Medium/Hard/Inter/Wet)
        df["start_compound"] = df["start_compound"].astype(str).str.strip().str.title()

        # optional: one-hots for LightGBM (or keep string and mark categorical in training)
        for comp in ["Soft","Medium","Hard","Inter","Wet"]:
            col = f"start_{comp.lower()}"
            df[col] = (df["start_compound"] == comp).astype("Int64")
    else:
        df["start_compound"] = np.nan
        for comp in ["Soft","Medium","Hard","Inter","Wet"]:
            df[f"start_{comp.lower()}"] = np.nan

    # Targets
    df["scored_points"] = (pd.to_numeric(df["finish_pos"], errors="coerce") <= 10).astype(int)

    # Save processed (full)
    proc_path = PROC_DIR / "driver_race_processed.parquet"
    df.to_parquet(proc_path, index=False)

    # ---- Final feature set for model ----
    # drop: relevance (if exists), track_overtake_idx/pit_loss_s (static), humidity already removed
    drop_cols = [c for c in ["relevance","track_overtake_idx","pit_loss_s"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    feature_cols = [
        "grid_pos","quali_gap_s","grid_quali_diff",
        "driver_last3_avg_finish","team_last3_avg_finish",
        "pos_change","race_total_overtakes",
        "pit_loss_total_s",
        "mean_air_temp","mean_track_temp","mean_wind_speed","mean_wind_dir","wind_sin","wind_cos",
        "is_wet_flag", "start_compound", "start_soft", "start_medium", "start_hard",
        "start_inter", "start_wet"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    keep = ["race_id","event_year","event_name","Driver","DriverNumber","TeamName",
            "finish_pos","scored_points"] + feature_cols
    out = df[keep].copy()

    # Minimal NA handling
    zero_fill = ["quali_gap_s","pit_loss_total_s","mean_air_temp","mean_track_temp",
                 "mean_wind_speed","mean_wind_dir","wind_sin","wind_cos","pos_change","total_overtakes"]
    for c in zero_fill:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    if "grid_pos" in out.columns:
        out = out.dropna(subset=["grid_pos"])

    # Save features
    out_path = FE_DIR / "standings_train.parquet"
    out.to_parquet(out_path, index=False)
    print(f"✅ Saved processed → {proc_path}")
    print(f"✅ Saved features  → {out_path} (rows={len(out)}, cols={out.shape[1]})")
    return out

if __name__ == "__main__":
    make_pre_race_table()
