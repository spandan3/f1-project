# backend/data/build_dataset.py
import numpy as np, pandas as pd
from collections import defaultdict, deque
from .utils import enable_cache, get_event_list, load_session_results, session_weather_snapshot, safe_int

MIN_FINISH_POS, MAX_FINISH_POS = 1, 20

def _rolling_mean(hist, k):
    if not hist:
        return 0.0
    return float(np.mean(hist[-k:])) if len(hist) >= k else float(np.mean(hist))

def build_dataset(seasons, cache_dir="./f1_cache"):
    """Return a feature dataframe: one row per driver per race with pre‑race features + targets."""
    enable_cache(cache_dir)
    rows = []

    for year in seasons:
        events = get_event_list(year)
        driver_hist = defaultdict(lambda: deque(maxlen=5))
        team_hist   = defaultdict(lambda: deque(maxlen=5))

        for rnd, ename, _ in events:
            q_ses, q_df = load_session_results(year, rnd, "Qualifying")
            quali_pos = {}
            if q_df is not None:
                for _, r in q_df.iterrows():
                    d = r.get("Abbreviation")
                    qp = safe_int(r.get("Position"))
                    if isinstance(d, str) and not np.isnan(qp):
                        quali_pos[d] = int(qp)

            r_ses, r_df = load_session_results(year, rnd, "Race")
            if r_ses is None or r_df is None or r_df.empty:
                continue

            weather = session_weather_snapshot(r_ses)

            for _, rr in r_df.iterrows():
                drv = rr.get("Abbreviation")
                team = rr.get("Team") if isinstance(rr.get("Team"), str) else "Unknown"
                finish_pos = safe_int(rr.get("Position"))
                grid = safe_int(rr.get("GridPosition"))
                pts = rr.get("Points")
                pts = float(pts) if pts is not None and not pd.isna(pts) else 0.0

                if not isinstance(drv, str) or drv.strip() == "":
                    continue
                if not (MIN_FINISH_POS <= (finish_pos or 999) <= MAX_FINISH_POS):
                    continue

                drv_hist = list(driver_hist[drv])
                team_hist_list = list(team_hist[team])

                row = {
                    "season": year,
                    "round": rnd,
                    "track_event_name": ename,
                    "driver": drv,
                    "team": team,
                    "quali_pos": quali_pos.get(drv, np.nan),
                    "grid_pos": grid if not np.isnan(grid if grid is not None else np.nan) else np.nan,
                    "driver_form_3": _rolling_mean(drv_hist, 3),
                    "driver_form_5": _rolling_mean(drv_hist, 5),
                    "team_form_3": _rolling_mean(team_hist_list, 3),
                    "team_form_5": _rolling_mean(team_hist_list, 5),
                    "air_temp": weather.get("AirTemp", np.nan),
                    "track_temp": weather.get("TrackTemp", np.nan),
                    "humidity": weather.get("Humidity", np.nan),
                    "wind_speed": weather.get("WindSpeed", np.nan),
                    "wind_dir": weather.get("WindDirection", np.nan),
                    "pressure": weather.get("Pressure", np.nan),
                    "rainfall": weather.get("Rainfall", np.nan),
                    "finish_pos": finish_pos,
                    "scored_points": 1 if pts > 0 else 0,
                }
                row["relevance"] = 21 - row["finish_pos"]
                rows.append(row)

            # update rolling history AFTER we created rows
            for _, rr in r_df.iterrows():
                d = rr.get("Abbreviation")
                t = rr.get("Team") if isinstance(rr.get("Team"), str) else None
                p = rr.get("Points"); p = float(p) if p is not None and not pd.isna(p) else 0.0
                if isinstance(d, str): driver_hist[d].append(p)
                if isinstance(t, str): team_hist[t].append(p)

    df = pd.DataFrame(rows)

    # light cleanup / encodings
    df["quali_pos"] = df["quali_pos"].fillna(30)
    df["grid_pos"]  = df["grid_pos"].fillna(30)
    df["track_id"]  = df["track_event_name"].astype("category").cat.codes
    df["team_id"]   = df["team"].astype("category").cat.codes
    df["driver_id"] = df["driver"].astype("category").cat.codes
    return df

def build_single_race_features(season: int, rnd: int, cache_dir="./f1_cache"):
    """Features for one race, to use after quali for predictions."""
    enable_cache(cache_dir)
    # Reuse logic by building minimal dataset only for that race, without targets
    from .utils import get_event_list
    events = {r for r,_,_ in get_event_list(season)}
    assert rnd in events, f"Round {rnd} not found for {season}"
    # A quick hack: build_dataset for season up to this round, then filter
    df = build_dataset([season], cache_dir)
    return df[(df["season"]==season) & (df["round"]==rnd)].drop(columns=["finish_pos","scored_points","relevance"])
