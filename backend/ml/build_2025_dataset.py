# build_2025_dataset.py
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import fastf1

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = BASE_DIR / "data" / "raw"
FE_DIR   = BASE_DIR / "data" / "fe" / "2025"
PROC_DIR = BASE_DIR / "data" / "processed"
for d in (RAW_DIR, FE_DIR, PROC_DIR, BASE_DIR / "f1_cache"):
    d.mkdir(parents=True, exist_ok=True)

# FastF1 disk cache
fastf1.Cache.enable_cache(str(BASE_DIR / "f1_cache"))

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
    
def _compute_pit_losses_sum(laps: pd.DataFrame, n_baseline: int = 3) -> pd.DataFrame:
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["event_year","event_name","DriverNumber","pit_loss_total_s"])

    for c in ["LapNumber","LapTime","PitInTime","PitOutTime","event_year","event_name","DriverNumber","Driver"]:
        if c not in laps.columns: laps[c] = np.nan

    def _sec(x):
        if pd.isna(x): return np.nan
        if isinstance(x, pd.Timedelta): return x.total_seconds()
        td = pd.to_timedelta(x, errors="coerce")
        if not pd.isna(td): return td.total_seconds()
        try: return float(x)
        except: return np.nan

    rows = []
    laps = laps.sort_values(["event_year","event_name","Driver","LapNumber"])
    for (ey,en,drv,dnum), g in laps.groupby(["event_year","event_name","Driver","DriverNumber"], sort=False):
        g = g.sort_values("LapNumber")
        out_idx = g.index[g["PitOutTime"].notna()].tolist()
        for oi in out_idx:
            out_row = g.loc[oi]; out_lap = int(out_row["LapNumber"])
            prev_row = g[g["LapNumber"] == out_lap - 1]
            if prev_row.empty: continue
            in_row = prev_row.iloc[0]

            prev_clean = g[(g["LapNumber"] < out_lap) & g["PitInTime"].isna() & g["PitOutTime"].isna() & g["LapTime"].notna()].tail(n_baseline)
            baseline = _sec(prev_clean["LapTime"].median()) if len(prev_clean) else _sec(g[g["LapTime"].notna()]["LapTime"].median())

            in_lt  = _sec(in_row["LapTime"])
            out_lt = _sec(out_row["LapTime"])
            loss = np.nan if any(np.isnan([baseline,in_lt,out_lt])) else (in_lt + out_lt - 2*baseline)
            rows.append({"event_year":ey,"event_name":en,"DriverNumber":dnum,"pit_loss_s":loss})

    per_stop = pd.DataFrame(rows)
    if per_stop.empty:
        return pd.DataFrame(columns=["event_year","event_name","DriverNumber","pit_loss_total_s"])
    return (per_stop.groupby(["event_year","event_name","DriverNumber"])["pit_loss_s"]
                  .sum().reset_index().rename(columns={"pit_loss_s":"pit_loss_total_s"}))


# ---------- RAW: fetch 2025 only ----------
def fetch_2025_raw():
    years = [2025]
    gprs = [
    "Melbourne",
    "Suzuka",
    "Shanghai",
    "Bahrain",
    "Jeddah",
    "Miami",
    "Monaco",
    "Spain",
    "Austria",
    "Belgium",
    "Silverstone",
    "Canada"
    ]

    laps_all, res_all, wx_all = [], [], []
    for y in years:
        cal = fastf1.get_event_schedule(y)
        events = cal["EventName"].tolist() if gprs is None else gprs
        for gp in events:
            for kind in ["Q", "R"]:
                try:
                    s = fastf1.get_session(y, gp, kind); s.load()
                    laps = s.laps.copy();      laps["event_year"]=y; laps["event_name"]=s.event["EventName"]; laps["session_type"]=kind
                    res  = s.results.copy();   res["event_year"]=y;  res["event_name"]=s.event["EventName"];  res["session_type"]=kind
                    wx   = s.weather_data.copy(); wx["event_year"]=y; wx["event_name"]=s.event["EventName"]; wx["session_type"]=kind
                    laps_all.append(laps); res_all.append(res); wx_all.append(wx)
                    print(f"OK: {y} {gp} {kind}")
                except Exception as e:
                    print(f"Skip: {y} {gp} {kind} -> {e}", file=sys.stderr)

    if laps_all:
        pd.concat(laps_all, ignore_index=True).to_parquet(RAW_DIR / "laps_2025.parquet", index=False)
    if res_all:
        pd.concat(res_all,  ignore_index=True).to_parquet(RAW_DIR / "results_2025.parquet", index=False)
    if wx_all:
        pd.concat(wx_all,   ignore_index=True).to_parquet(RAW_DIR / "weather_2025.parquet", index=False)
    print(f"✅ Saved raw to {RAW_DIR} (suffix _2025)")

# ---------- Weather aggregation ----------
def build_weather_features(raw_wx_path: Path) -> pd.DataFrame | None:
    if not raw_wx_path.exists(): 
        return None
    wx = pd.read_parquet(raw_wx_path).copy()
    st_col = _first_col(wx, ["session_type","SessionType"], None)
    if st_col is None:
        return None
    wx_r = wx[wx[st_col].astype(str) == "R"].copy()
    ycol = _first_col(wx_r, ["event_year","EventYear"], "event_year")
    ncol = _first_col(wx_r, ["event_name","EventName"], "event_name")

    agg = {}
    if "AirTemp" in wx_r:       agg["AirTemp"] = "mean"
    if "TrackTemp" in wx_r:     agg["TrackTemp"] = "mean"
    if "Humidity" in wx_r:      agg["Humidity"] = "mean"
    if "Rainfall" in wx_r:      agg["Rainfall"] = "max"
    if "WindSpeed" in wx_r:     agg["WindSpeed"] = "mean"
    if "WindDirection" in wx_r: agg["WindDirection"] = "mean"
    if not agg: return None

    wx_feat = (wx_r.groupby([ycol, ncol]).agg(agg).reset_index()
                  .rename(columns={ycol:"event_year", ncol:"event_name",
                                   "AirTemp":"mean_air_temp",
                                   "TrackTemp":"mean_track_temp",
                                   "Humidity":"mean_humidity",
                                   "Rainfall":"rain_any",
                                   "WindSpeed":"mean_wind_speed",
                                   "WindDirection":"mean_wind_dir"}))
    wx_feat["rain_any"] = wx_feat.get("rain_any", 0).fillna(0)
    wx_feat["is_wet_flag"] = (wx_feat["rain_any"] > 0).astype(int)

    if "mean_wind_dir" in wx_feat:
        rad = np.deg2rad(wx_feat["mean_wind_dir"])
        wx_feat["wind_sin"] = np.sin(rad)
        wx_feat["wind_cos"] = np.cos(rad)
    else:
        wx_feat["wind_sin"] = np.nan
        wx_feat["wind_cos"] = np.nan
    return wx_feat

# ---------- Build features from raws ----------
def build_2025_features():
    RAW_LAPS = RAW_DIR / "laps_2025.parquet"
    RAW_RES  = RAW_DIR / "results_2025.parquet"
    RAW_WX   = RAW_DIR / "weather_2025.parquet"

    if not RAW_RES.exists():
        raise FileNotFoundError("results_2025.parquet not found. Run fetch first.")

    res  = pd.read_parquet(RAW_RES)
    laps = pd.read_parquet(RAW_LAPS) if RAW_LAPS.exists() else pd.DataFrame()

    st_col  = _first_col(res, ["session_type","SessionType"], None)
    if st_col is None:
        raise ValueError("results_2025.parquet has no session type column")

    drv_col  = _first_col(res, ["Driver","Abbreviation","DriverId"], "Driver")
    team_col = _first_col(res, ["TeamName","Team"], "TeamName")
    grid_col = _first_col(res, ["GridPosition"], "GridPosition")
    pos_col  = _first_col(res, ["Position"], "Position")
    stat_col = _first_col(res, ["Status"], "Status")
    year_col = _first_col(res, ["event_year","EventYear"], "event_year")
    name_col = _first_col(res, ["event_name","EventName"], "event_name")
    dnum_col = _first_col(res, ["DriverNumber"], "DriverNumber")

    # Qualifying slice
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

    # Race slice
    race = res[res[st_col].astype(str) == "R"].copy().rename(columns={
        drv_col:"Driver", team_col:"TeamName",
        year_col:"event_year", name_col:"event_name",
        dnum_col:"DriverNumber", pos_col:"finish_pos",
        stat_col:"Status", grid_col:"GridPosition"
    })

    # Merge
    df = pd.merge(
        race[["DriverNumber","Driver","TeamName","event_year","event_name","finish_pos","Status","GridPosition"]],
        quali_feat,
        on=["DriverNumber","event_year","event_name"],
        how="left"
    )
    df["grid_pos"] = df["GridPosition"].where(df["GridPosition"].notna(), df["quali_pos_final"])
    df["grid_quali_diff"] = df["grid_pos"] - df["quali_pos_final"]
    df = df.drop(columns=["GridPosition"])

    # Sort for rolling
    df = df.sort_values(["Driver","event_year","event_name"], kind="mergesort").reset_index(drop=True)

    # Rolling averages
    last3 = lambda s: s.shift().rolling(3, min_periods=1).mean()
    df["driver_last3_avg_finish"] = df.groupby("Driver", group_keys=False)["finish_pos"].transform(last3)
    df["team_last3_avg_finish"]   = df.groupby("TeamName", group_keys=False)["finish_pos"].transform(last3)

    # Fill early-season gaps
    for c in ["driver_last3_avg_finish","team_last3_avg_finish"]:
        key = "Driver" if c.startswith("driver") else "TeamName"
        df[c] = df.groupby(key)[c].transform(lambda s: s.fillna(s.mean()))
        df[c] = df[c].fillna(df[c].mean())

    # Position change + race id
    df["pos_change"] = df["grid_pos"] - df["finish_pos"]
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)

    # after you create df["pos_change"] and df["race_id"]
    race_agg = (df.assign(_gain=df["pos_change"].clip(lower=0))
                .groupby("race_id")["_gain"]
                .sum()
                .rename("race_total_overtakes")
                .reset_index())
    df = df.merge(race_agg, on="race_id", how="left")




    # Lap-based overtakes (simple)
    if not laps.empty and all(c in laps.columns for c in ["event_year","event_name","Driver","LapNumber","Position"]):
        laps = laps.sort_values(["event_year","event_name","Driver","LapNumber"])
        pos_diff = laps.groupby(["event_year","event_name","Driver"])["Position"].diff()
        drv_ov = (pos_diff < 0).groupby([laps["event_year"], laps["event_name"], laps["Driver"]]).sum().reset_index(name="driver_overtakes")
        drv_be = (pos_diff > 0).groupby([laps["event_year"], laps["event_name"], laps["Driver"]]).sum().reset_index(name="driver_times_overtaken")
        drv = drv_ov.merge(drv_be, on=["event_year","event_name","Driver"], how="outer").fillna(0)
        drv["driver_net_passes"] = drv["driver_overtakes"] - drv["driver_times_overtaken"]
        df = df.merge(drv, on=["event_year","event_name","Driver"], how="left")
    else:
        df["driver_overtakes"] = np.nan
        df["driver_times_overtaken"] = np.nan
        df["driver_net_passes"] = np.nan

    # Weather merge
    wx_feat = build_weather_features(RAW_WX)
    if wx_feat is not None:
        df = df.merge(wx_feat, on=["event_year","event_name"], how="left")
        df["is_wet_flag"] = df.get("is_wet_flag", 0).fillna(0).astype(int)
        if "mean_humidity" in df:  # we can drop if not needed
            df = df.drop(columns=["mean_humidity"])
    else:
        for c in ["mean_air_temp","mean_track_temp","mean_wind_speed","mean_wind_dir","wind_sin","wind_cos"]:
            df[c] = np.nan
        df["is_wet_flag"] = 0

    # Starting compound (from first lap with compound)
    if not laps.empty:
        for c in ["event_year","event_name","Driver","LapNumber","Compound"]:
            if c not in laps.columns: laps[c] = np.nan
        tyre_first = (laps.sort_values(["event_year","event_name","Driver","LapNumber"])
                           .dropna(subset=["Compound"])
                           .groupby(["event_year","event_name","Driver"], as_index=False)
                           .first()[["event_year","event_name","Driver","Compound"]]
                           .rename(columns={"Compound":"start_compound"}))  
        df = df.merge(tyre_first, on=["event_year","event_name","Driver"], how="left")
        df["start_compound"] = df["start_compound"].astype(str).str.strip().str.title()
        pit_tot = _compute_pit_losses_sum(laps, n_baseline=3)
        df = df.merge(pit_tot, on=["event_year","event_name","DriverNumber"], how="left")
        for comp in ["Soft","Medium","Hard","Inter","Wet"]:
            df[f"start_{comp.lower()}"] = (df["start_compound"] == comp).astype("Int64")
    else:
        df["start_compound"] = np.nan
        df["pit_loss_total_s"] = np.nan
        for comp in ["Soft","Medium","Hard","Inter","Wet"]:
            df[f"start_{comp.lower()}"] = np.nan

    # Quick “scored points” label (optional for classification)
    df["scored_points"] = (pd.to_numeric(df["finish_pos"], errors="coerce") <= 10).astype(int)

    # Save processed full table (debugging/EDA)
    proc_path = PROC_DIR / "driver_race_processed_2025.parquet"
    df.to_parquet(proc_path, index=False)

    # Final feature set (aligns with typical training)
    feature_cols = [
    "grid_pos","quali_gap_s","grid_quali_diff",
    "driver_last3_avg_finish","team_last3_avg_finish",
    "pos_change","race_total_overtakes",
    "driver_overtakes","driver_times_overtaken","driver_net_passes",
    "pit_loss_total_s",
    "mean_air_temp","mean_track_temp","mean_wind_speed","mean_wind_dir","wind_sin","wind_cos",
    "is_wet_flag","start_compound","start_soft","start_medium","start_hard","start_inter","start_wet"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    keep = ["race_id","event_year","event_name","Driver","DriverNumber","TeamName",
            "finish_pos","scored_points"] + feature_cols
    out = df[keep].copy()

    # Simple NA handling
    zero_fill = ["quali_gap_s","mean_air_temp","mean_track_temp","mean_wind_speed","mean_wind_dir","wind_sin","wind_cos",
                 "pos_change","driver_overtakes","driver_times_overtaken","driver_net_passes"]
    for c in zero_fill:
        if c in out.columns: out[c] = out[c].fillna(0.0)
    if "grid_pos" in out.columns:
        out = out.dropna(subset=["grid_pos"])

    fe_path = FE_DIR / "standings_2025.parquet"
    out.to_parquet(fe_path, index=False)
    print(f"✅ Saved processed → {proc_path}")
    print(f"✅ Saved features  → {fe_path} (rows={len(out)}, cols={out.shape[1]})")
    return fe_path

# ---------- Orchestrate ----------
if __name__ == "__main__":
    print("▶ Fetching 2025 raw…")
    fetch_2025_raw()
    print("▶ Building 2025 features…")
    build_2025_features()
    print("✅ Done.")
