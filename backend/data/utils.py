# backend/data/utils.py
import os, math, numpy as np, pandas as pd
import fastf1

CACHE_DIR = os.getenv("F1_CACHE_DIR", "./f1_cache")

def enable_cache(cache_dir: str = CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def safe_int(x):
    try:
        v = int(x)
        return v if not math.isnan(v) else np.nan
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return np.nan

def get_event_list(year: int):
    ev = fastf1.get_event_schedule(year, include_testing=False).sort_values("EventDate")
    out = []
    for _, r in ev.iterrows():
        rnd = safe_int(r.get("RoundNumber"))
        if not np.isnan(rnd):
            out.append((int(rnd), r.get("EventName"), pd.to_datetime(r.get("EventDate"))))
    return out

def load_session_results(year: int, round_number: int, kind: str):
    """Return (session, results_df) for 'Race' or 'Qualifying'. Results df has Abbreviation, Team, Position, GridPosition, Points, Status."""
    try:
        ses = fastf1.get_session(year, round_number, kind)
        ses.load()
        df = ses.results.copy()
        if df is None or df.empty:
            return None, None
        if "Team" not in df.columns and "TeamName" in df.columns:
            df = df.rename(columns={"TeamName": "Team"})
        for c in ["Abbreviation", "DriverNumber", "Team"]:
            if c not in df.columns:
                df[c] = np.nan
        if kind == "Race":
            for c in ["Position", "GridPosition", "Points", "Status"]:
                if c not in df.columns:
                    df[c] = np.nan
        if kind == "Qualifying" and "Position" not in df.columns:
            df["Position"] = np.nan
        return ses, df
    except Exception:
        return None, None

def session_weather_snapshot(session):
    """Simple pre‑race weather snapshot from first row."""
    try:
        w = session.weather_data
        if w is None or w.empty:
            return {}
        row = w.iloc[0]
        fields = ["AirTemp","TrackTemp","Humidity","WindSpeed","WindDirection","Pressure","Rainfall"]
        return {f: float(row.get(f, np.nan)) for f in fields}
    except Exception:
        return {}
