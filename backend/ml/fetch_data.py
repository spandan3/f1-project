# backend/ml/fetch_data.py
"""
Fetch raw F1 data from FastF1 API.
Supports fetching multiple years for training or single races for updates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fastf1
import pandas as pd

from .common import RAW_DIR, CACHE_DIR

# Enable FastF1 on-disk cache
fastf1.Cache.enable_cache(str(CACHE_DIR))


def fetch_years(start_year: int, end_year: int, events: list[str] | None = None) -> dict:
    """
    Fetch raw F1 data for a range of years.
    
    Args:
        start_year: First year to fetch (e.g., 2015)
        end_year: Last year to fetch (e.g., 2025)
        events: Optional list of specific GP names to fetch (default: all)
        
    Returns:
        Dict with counts of rows fetched
    """
    laps_all, res_all, wx_all = [], [], []
    
    for year in range(start_year, end_year + 1):
        try:
            cal = fastf1.get_event_schedule(year)
            event_names = events if events else cal["EventName"].tolist()
        except Exception as e:
            print(f"⚠️ Could not get {year} schedule: {e}")
            continue
        
        print(f"\n📅 {year}: {len(event_names)} events")
        
        for gp in event_names:
            for kind in ["Q", "R"]:
                try:
                    s = fastf1.get_session(year, gp, kind)
                    s.load()
                    
                    # Laps
                    laps = s.laps.copy()
                    laps["event_year"] = year
                    laps["event_name"] = s.event["EventName"]
                    laps["session_type"] = kind
                    laps_all.append(laps)
                    
                    # Results
                    res = s.results.copy()
                    res["event_year"] = year
                    res["event_name"] = s.event["EventName"]
                    res["session_type"] = kind
                    res_all.append(res)
                    
                    # Weather
                    wx = s.weather_data.copy()
                    wx["event_year"] = year
                    wx["event_name"] = s.event["EventName"]
                    wx["session_type"] = kind
                    wx_all.append(wx)
                    
                    print(f"   ✓ {gp} {kind}")
                    
                except Exception as e:
                    print(f"   ✗ {gp} {kind}: {e}", file=sys.stderr)

    # Save to parquet
    stats = {"laps": 0, "results": 0, "weather": 0}
    
    if laps_all:
        df = pd.concat(laps_all, ignore_index=True)
        df.to_parquet(RAW_DIR / "laps.parquet", index=False)
        stats["laps"] = len(df)
        print(f"\n✅ Saved laps: {len(df):,} rows")
        
    if res_all:
        df = pd.concat(res_all, ignore_index=True)
        df.to_parquet(RAW_DIR / "results.parquet", index=False)
        stats["results"] = len(df)
        print(f"✅ Saved results: {len(df):,} rows")
        
    if wx_all:
        df = pd.concat(wx_all, ignore_index=True)
        df.to_parquet(RAW_DIR / "weather.parquet", index=False)
        stats["weather"] = len(df)
        print(f"✅ Saved weather: {len(df):,} rows")

    print(f"\n📁 Raw data saved to {RAW_DIR}")
    return stats


def fetch_single_race(year: int, event_name: str) -> dict:
    """
    Fetch a single race and APPEND to existing data files.
    Used for rolling updates after each 2026 race.
    
    Args:
        year: Race year
        event_name: Exact GP name (e.g., "Bahrain Grand Prix")
        
    Returns:
        Dict with status
    """
    print(f"\n🏎️ Fetching {year} {event_name}...")
    
    new_laps, new_res, new_wx = [], [], []
    
    for kind in ["Q", "R"]:
        try:
            s = fastf1.get_session(year, event_name, kind)
            s.load()
            
            # Laps
            laps = s.laps.copy()
            laps["event_year"] = year
            laps["event_name"] = s.event["EventName"]
            laps["session_type"] = kind
            new_laps.append(laps)
            
            # Results
            res = s.results.copy()
            res["event_year"] = year
            res["event_name"] = s.event["EventName"]
            res["session_type"] = kind
            new_res.append(res)
            
            # Weather
            wx = s.weather_data.copy()
            wx["event_year"] = year
            wx["event_name"] = s.event["EventName"]
            wx["session_type"] = kind
            new_wx.append(wx)
            
            print(f"   ✓ {kind} session loaded")
            
        except Exception as e:
            print(f"   ✗ {kind}: {e}", file=sys.stderr)
            return {"ok": False, "error": str(e)}

    # Append to existing files
    laps_path = RAW_DIR / "laps.parquet"
    res_path = RAW_DIR / "results.parquet"
    wx_path = RAW_DIR / "weather.parquet"
    
    # Helper to append
    def append_parquet(path: Path, new_data: list):
        if new_data:
            new_df = pd.concat(new_data, ignore_index=True)
            if path.exists():
                existing = pd.read_parquet(path)
                # Remove any existing rows for this race (in case of re-fetch)
                mask = ~((existing["event_year"] == year) & 
                        (existing["event_name"] == event_name))
                existing = existing[mask]
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(path, index=False)
            return len(new_df)
        return 0
    
    laps_added = append_parquet(laps_path, new_laps)
    res_added = append_parquet(res_path, new_res)
    wx_added = append_parquet(wx_path, new_wx)
    
    print(f"✅ Added {res_added} result rows, {laps_added} lap rows, {wx_added} weather rows")
    
    return {
        "ok": True,
        "year": year,
        "event": event_name,
        "laps_added": laps_added,
        "results_added": res_added,
        "weather_added": wx_added
    }


def get_available_races(year: int) -> list[dict]:
    """Get list of races for a year with round, name, and date."""
    try:
        cal = fastf1.get_event_schedule(year)
        races = []
        for _, row in cal.iterrows():
            event_name = row["EventName"]
            # Filter out testing and non-race events
            if "Testing" in event_name or "Test" in event_name:
                continue
            
            round_num = row.get("RoundNumber")
            # Handle NaN or missing round numbers
            if pd.isna(round_num):
                round_num = len(races) + 1
            else:
                round_num = int(round_num)
            
            # Only include if it's a valid round (>= 1) or if RoundNumber is missing but it's a GP
            if round_num >= 1 or "Grand Prix" in event_name:
                races.append({
                    "round": round_num if round_num >= 1 else len(races) + 1,
                    "race_name": event_name,
                    "date": str(row.get("EventDate", "")) if pd.notna(row.get("EventDate")) else "",
                })
        return races
    except Exception as e:
        print(f"⚠️ Error getting races for {year}: {e}", file=sys.stderr)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch F1 data from FastF1 API")
    parser.add_argument(
        "--years", type=str, default="2015-2025",
        help="Year range to fetch (e.g., '2015-2025' or '2024')"
    )
    parser.add_argument(
        "--race", type=str, default=None,
        help="Fetch single race (e.g., '2026_Bahrain Grand Prix')"
    )
    args = parser.parse_args()
    
    if args.race:
        # Single race mode
        parts = args.race.split("_", 1)
        if len(parts) != 2:
            print("Error: Use format 'YEAR_Event Name' (e.g., '2026_Bahrain Grand Prix')")
            sys.exit(1)
        year, event = int(parts[0]), parts[1]
        fetch_single_race(year, event)
    else:
        # Year range mode
        if "-" in args.years:
            start, end = map(int, args.years.split("-"))
        else:
            start = end = int(args.years)
        fetch_years(start, end)

