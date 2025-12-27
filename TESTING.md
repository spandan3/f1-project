# 🧪 Testing Guide

Complete guide to test the F1 Prediction System end-to-end.

---

## 📋 Prerequisites

1. **Backend is running:**
   ```bash
   python run_api.py
   ```
   API should be at `http://localhost:8000`

2. **Frontend is running:**
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend should be at `http://localhost:5173`

---

## 🧪 Test 1: Backend API Health Check

### Test the root endpoint:
```bash
curl http://localhost:8000/
```

**Expected Response:**
```json
{
  "status": "ok",
  "service": "F1 Predictions API",
  "version": "2.0.0"
}
```

### Test status endpoint:
```bash
curl http://localhost:8000/status
```

**Expected Response:**
```json
{
  "available_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  "model_loaded": true,
  "total_races": 173
}
```

---

## 🧪 Test 2: List Available Races

### Get races for a specific year:
```bash
curl http://localhost:8000/races/2025
```

**Expected Response:**
```json
[
  {
    "round": 1,
    "race_name": "Bahrain Grand Prix",
    "date": "2025-03-02"
  },
  {
    "round": 2,
    "race_name": "Saudi Arabian Grand Prix",
    "date": "2025-03-09"
  },
  ...
]
```

---

## 🧪 Test 3: Generate Predictions (CLI)

### Test predictions for a specific year:
```bash
python -m backend.ml.predict --year 2025
```

**Expected Output:**
- Predictions CSV saved to `data/preds/predictions_2025.csv`
- Metrics CSV saved to `data/preds/metrics_2025.csv`
- Console output showing NDCG@3, NDCG@10, Top-3 Hit, Spearman Rho

### Test predictions for a specific race:
```bash
python -m backend.ml.predict --race-id "2025_Bahrain Grand Prix"
```

**Expected Output:**
- Predictions for that specific race
- Metrics if actual results are available

---

## 🧪 Test 4: Generate Predictions (API)

### Test via API with year and round:
```bash
curl "http://localhost:8000/predict?year=2025&round=1"
```

**Expected Response:**
```json
{
  "year": 2025,
  "round": 1,
  "race_name": "Bahrain Grand Prix",
  "predictions": [
    {
      "pred_pos": 1,
      "driver": "Max Verstappen",
      "constructor": "Red Bull Racing",
      "grid_pos": 1,
      "pred_score": 0.95,
      "actual_pos": 1,
      "finish_pos": 1
    },
    ...
  ],
  "metrics": {
    "ndcg_3": 0.86,
    "ndcg_10": 0.89,
    "top3_hit": 0.67,
    "spearman_rho": 0.74
  }
}
```

### Test with just year (all races):
```bash
curl "http://localhost:8000/predict?year=2025"
```

---

## 🧪 Test 5: Frontend Testing

### 5.1 Landing Page (`/`)
1. Open `http://localhost:5173`
2. **Check:**
   - ✅ Page loads without errors
   - ✅ System status shows available years
   - ✅ Year selector works
   - ✅ Race selector populates when year is selected
   - ✅ "Generate Prediction" button is clickable

### 5.2 Dashboard (`/dashboard`)
1. Select a year (e.g., 2025) and race (e.g., "Bahrain Grand Prix")
2. Click "Generate Prediction"
3. **Check:**
   - ✅ Loading spinner appears
   - ✅ Predictions table displays with:
     - Predicted position
     - Driver name
     - Team/Constructor
     - Grid position
     - Predicted score
   - ✅ Podium card shows top 3 drivers correctly (P1 gold, P2 silver, P3 bronze)
   - ✅ Metrics display shows:
     - NDCG@3
     - NDCG@10
     - Top-3 Hit
     - Spearman Rho
   - ✅ Charts render (Grid vs Predicted, Position Delta)
   - ✅ Comparison table shows predicted vs actual (if available)

### 5.3 Upcoming Races (`/upcoming`)
1. Navigate to `/upcoming`
2. **Check:**
   - ✅ List of upcoming races displays
   - ✅ Can generate predictions for future races
   - ✅ "Pre-race prediction" labels appear

### 5.4 Chat Page (`/chat`)
1. Navigate to `/chat`
2. **Check:**
   - ✅ Page loads with "Coming Soon" banner
   - ✅ Chat interface displays
   - ✅ Can type messages (but gets placeholder response)

---

## 🧪 Test 6: Full Workflow Test (Simulate 2026 Race)

This simulates the complete workflow for a 2026 race.

### Step 1: Fetch Data (Saturday - After Qualifying)
```bash
# This would fetch 2026 data (when available)
python -m backend.ml.fetch_data --years 2026-2026
```

**Expected:**
- Raw data saved to `data/raw/`
- Qualifying results available

### Step 2: Build Pre-Race Features
```bash
python -m backend.ml.build_inference_rows --year 2026 --event "Bahrain Grand Prix"
```

**Expected:**
- Features file created: `data/fe/inference/infer_2026_Bahrain_Grand_Prix.parquet`
- Only pre-race features included

### Step 3: Generate Prediction
```bash
python -m backend.ml.predict --fe-path data/fe/inference/infer_2026_Bahrain_Grand_Prix.parquet
```

**Expected:**
- Predictions CSV with predicted finishing order
- No metrics (race hasn't happened yet)

### Step 4: Update After Race (Sunday - After Race)
```bash
python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix"
```

**Expected:**
- Fetches race results
- Rebuilds features with actual results
- Retrains model with new data
- Model performance improves

---

## 🧪 Test 7: Model Training

### Test quick training:
```bash
python -m backend.ml.trainer
```

**Expected:**
- Model trains in ~1-2 minutes
- Model saved to `models/ranker_lgb.txt`
- Metadata saved to `models/ranker_meta.joblib`
- Console shows training metrics

### Test with hyperparameter tuning:
```bash
python -m backend.ml.trainer --tune --n-trials 20
```

**Expected:**
- Optuna optimization runs (~10-30 min)
- Best parameters found
- Model trained with optimized parameters

---

## 🧪 Test 8: Error Handling

### Test invalid year:
```bash
curl "http://localhost:8000/predict?year=2010"
```

**Expected:** 404 or 400 error with message

### Test invalid round:
```bash
curl "http://localhost:8000/predict?year=2025&round=99"
```

**Expected:** 404 error - "Race round 99 not found for year 2025"

### Test missing model:
```bash
# Temporarily rename model file
mv models/ranker_lgb.txt models/ranker_lgb.txt.bak
curl "http://localhost:8000/predict?year=2025&round=1"
# Restore it
mv models/ranker_lgb.txt.bak models/ranker_lgb.txt
```

**Expected:** 500 error with "Model not found" message

---

## 🧪 Test 9: API Documentation

### Open Swagger UI:
1. Navigate to `http://localhost:8000/docs`
2. **Check:**
   - ✅ All endpoints are listed
   - ✅ Can test endpoints directly from UI
   - ✅ Request/response schemas are shown

---

## ✅ Checklist: All Tests Pass

- [ ] Backend API health check works
- [ ] Status endpoint returns correct data
- [ ] Race listing works for multiple years
- [ ] CLI predictions generate correctly
- [ ] API predictions return valid JSON
- [ ] Frontend landing page loads
- [ ] Frontend dashboard displays predictions
- [ ] Frontend charts render correctly
- [ ] Podium display shows correct positions (P1 gold, P2 silver, P3 bronze)
- [ ] Metrics display correctly
- [ ] Error handling works for invalid inputs
- [ ] Model training completes successfully
- [ ] Full workflow (fetch → build → predict → update) works

---

## 🐛 Troubleshooting

### "Model not found" error
```bash
python -m backend.ml.trainer
```

### "Features file not found" error
```bash
python -m backend.ml.build_features
```

### Frontend shows "Failed to fetch"
- Check backend is running on `http://localhost:8000`
- Check CORS is configured correctly
- Check browser console for errors

### Predictions show all zeros or NaN
- Verify features file has data: `python -c "import pandas as pd; print(pd.read_parquet('data/fe/features.parquet').shape)"`
- Rebuild features: `python -m backend.ml.build_features`
- Retrain model: `python -m backend.ml.trainer`

---

## 📊 Expected Performance Metrics

When testing on 2025 data, you should see metrics like:

- **NDCG@3**: ~0.85-0.90 (podium prediction quality)
- **NDCG@10**: ~0.88-0.92 (points positions)
- **Top-3 Hit**: ~0.60-0.70 (% of actual podium in predicted top 3)
- **Spearman ρ**: ~0.70-0.80 (overall rank correlation)

These may vary slightly based on the specific races tested.

