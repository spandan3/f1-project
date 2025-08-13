from pathlib import Path
import fastf1, pandas as pd, sys

# Project root (two levels up from backend/data/)
BASE_DIR = Path(__file__).resolve().parents[2]

# FastF1 on-disk cache at <project_root>/f1_cache
fastf1.Cache.enable_cache(str(BASE_DIR / "f1_cache"))

RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2025, 2024, 2023, 2022, 2021, 2020] 
GPRS  = None # None = all events in season

def load_session(year, gp_name, kind):  
    s = fastf1.get_session(year, gp_name, kind)
    s.load()
    laps = s.laps.copy()
    laps["event_year"] = year
    laps["event_name"] = s.event["EventName"]
    laps["session_type"] = kind

    res = s.results.copy()
    res["event_year"] = year
    res["event_name"] = s.event["EventName"]
    res["session_type"] = kind

    wx = s.weather_data.copy()
    wx["event_year"] = year
    wx["event_name"] = s.event["EventName"]
    wx["session_type"] = kind

    return laps, res, wx

def main():
    laps_all, res_all, wx_all = [], [], []
    for y in YEARS:
        cal = fastf1.get_event_schedule(y)
        events = cal["EventName"].tolist() if GPRS is None else GPRS
        for gp in events:
            for kind in ["Q", "R"]:
                try:
                    laps, res, wx = load_session(y, gp, kind)
                    laps_all.append(laps); res_all.append(res); wx_all.append(wx)
                    print(f"OK: {y} {gp} {kind}")
                except Exception as e:
                    print(f"Skip: {y} {gp} {kind} -> {e}", file=sys.stderr)

    if laps_all:
        pd.concat(laps_all, ignore_index=True).to_parquet(RAW_DIR / "laps.parquet", index=False)
    if res_all:
        pd.concat(res_all, ignore_index=True).to_parquet(RAW_DIR / "results.parquet", index=False)
    if wx_all:
        pd.concat(wx_all, ignore_index=True).to_parquet(RAW_DIR / "weather.parquet", index=False)

    print(f"✅ Saved to {RAW_DIR}")

if __name__ == "__main__":
    main()
