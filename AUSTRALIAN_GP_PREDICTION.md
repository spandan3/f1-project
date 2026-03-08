# Australian Grand Prix Prediction - Quick Guide

## What Was Added

### Backend Changes

1. **New `/prepare-race` endpoint** (`backend/api.py`)
   - Fetches qualifying and race data from FastF1
   - Builds inference features automatically
   - Returns status of data fetch and feature building

2. **Enhanced `/predict` endpoint**
   - Already had logic to auto-build inference features for 2026 races
   - Now works seamlessly with the prepare endpoint

### Frontend Changes

1. **New `prepareRace()` method** (`frontend/src/lib/api.ts`)
   - Calls the `/prepare-race` endpoint
   - Handles race preparation before prediction

2. **Updated Upcoming Page** (`frontend/src/pages/Upcoming.tsx`)
   - Auto-selects next upcoming race (Australian GP if it's next)
   - Automatically prepares race data before generating predictions
   - Shows loading states for preparation and prediction
   - Better error handling

## How to Use

### Option 1: Via Frontend (Recommended)

1. **Start the backend:**
   ```bash
   python run_api.py
   ```

2. **Start the frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Navigate to Upcoming tab:**
   - The page will auto-select the Australian Grand Prix if it's the next race
   - Click "Prepare & Generate Prediction"
   - The system will:
     - Fetch qualifying data from FastF1
     - Build inference features
     - Generate predictions
     - Display results

### Option 2: Via API Directly

```bash
# 1. Prepare the race (fetch data + build features)
curl -X POST "http://localhost:8000/prepare-race?year=2026&race=Australian%20Grand%20Prix&build_features=true"

# 2. Get predictions
curl "http://localhost:8000/predict?year=2026&round=3"
```

### Option 3: Manual CLI (if needed)

If the API fails, you can still use the CLI:

```bash
# Fetch qualifying data
python -m backend.ml.fetch_data --race "2026_Australian Grand Prix"

# Build features
python -m backend.ml.build_inference_rows --year 2026 --event "Australian Grand Prix"

# Get predictions
python -m backend.ml.predict --fe-path data/fe/inference/infer_2026_Australian_Grand_Prix.parquet
```

## Files Modified

### Backend
- `backend/api.py` - Added `/prepare-race` endpoint

### Frontend
- `frontend/src/lib/api.ts` - Added `prepareRace()` method
- `frontend/src/pages/Upcoming.tsx` - Auto-prepare race, auto-select next race

## Data Flow

```
User clicks "Prepare & Generate Prediction"
    ↓
Frontend calls /prepare-race
    ↓
Backend fetches qualifying data from FastF1
    ↓
Backend builds inference features
    ↓
Frontend calls /predict
    ↓
Backend generates predictions using ML model
    ↓
Frontend displays predictions
```

## Notes

- **Qualifying data must be available**: The system needs qualifying results from FastF1
- **First time may be slow**: Fetching data from FastF1 can take 30-60 seconds
- **Subsequent calls are fast**: FastF1 caches data locally
- **Auto-selection**: The page automatically selects the next upcoming race

## Troubleshooting

**"Failed to prepare race"**
- Check that qualifying has completed (FastF1 needs session data)
- Verify internet connection (FastF1 fetches from F1 servers)
- Check FastF1 cache: `f1_cache/` directory

**"No races available for 2026"**
- The race calendar may not be loaded yet
- Try refreshing the page
- Check that FastF1 can access 2026 schedule

**Predictions not showing**
- Check browser console for errors
- Verify backend is running on port 8000
- Check backend logs for errors

