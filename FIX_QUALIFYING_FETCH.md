# Fix: Qualifying Data Fetch Error

## Problem
Error: "The data you are trying to access has not been loaded yet. See `Session.load`"

## Root Cause
The `fetch_single_race` function was trying to fetch both qualifying (Q) and race (R) sessions. For pre-race predictions, we only need qualifying data, and the race session doesn't exist yet.

## Solution Applied

### 1. Updated `fetch_single_race()` function
- Added `require_race` parameter (default: False)
- For pre-race predictions: Only fetches qualifying (Q)
- For post-race updates: Fetches both Q and R
- Better error handling for missing sessions

### 2. Updated `/prepare-race` endpoint
- Calls `fetch_single_race(year, race, require_race=False)`
- Only fetches qualifying data needed for predictions
- Better error messages explaining what went wrong

### 3. Updated `update.py`
- Calls `fetch_single_race(year, race, require_race=True)`
- After race completes, needs both sessions

## How to Use

### For Pre-Race Predictions (Australian GP)
```bash
# Via API - automatically only fetches qualifying
curl -X POST "http://localhost:8000/prepare-race?year=2026&race=Australian%20Grand%20Prix"
```

### For Post-Race Updates
```bash
# Via API - fetches both qualifying and race
curl -X POST "http://localhost:8000/update?year=2026&race=Australian%20Grand%20Prix"
```

## Troubleshooting

**Still getting "not been loaded" error?**
1. **Check if qualifying has completed**: FastF1 needs the session to be finished
2. **Wait a few minutes**: F1 servers may take 5-10 minutes to publish data
3. **Check FastF1 cache**: Look in `f1_cache/` directory
4. **Verify event name**: Must match exactly (e.g., "Australian Grand Prix" not "Australia GP")

**Test if session exists:**
```python
import fastf1
fastf1.Cache.enable_cache('./f1_cache')
s = fastf1.get_session(2026, "Australian Grand Prix", "Q")
s.load()  # This will fail if data isn't available
print(s.results)
```

## Files Modified
- `backend/ml/fetch_data.py` - Added `require_race` parameter
- `backend/api.py` - Pass `require_race=False` for pre-race
- `backend/ml/update.py` - Pass `require_race=True` for post-race

