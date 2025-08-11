import fastf1, pandas as pd, os, sys
fastf1.Cache.enable_cache("f1_cache")

YEARS = [2022, 2023]   # expand later
GPRS  = None           # None = all events in season

def load_session(year, gp_name, kind):
    s = fastf1.get_session(year, gp_name, kind); s.load()
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
    os.makedirs("data/raw", exist_ok=True)
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
    pd.concat(laps_all).to_parquet("data/raw/laps.parquet")
    pd.concat(res_all).to_parquet("data/raw/results.parquet")
    pd.concat(wx_all).to_parquet("data/raw/weather.parquet")

if __name__ == "__main__":
    main()
