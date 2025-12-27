# 🏎️ F1 Race Prediction System

A machine learning system that predicts Formula 1 race finishing order. Built for **rolling predictions** throughout the 2026 season.

---

## 📊 Current Status

| Component | Status |
|-----------|--------|
| Data Pipeline | ✅ Ready (2018-2025) |
| Feature Engineering | ✅ Ready (173 races) |
| LightGBM Ranking Model | ✅ Ready |
| Optuna Hyperparameter Tuning | ✅ Ready |
| Rolling Update System | ✅ Ready |
| REST API | ✅ Ready |
| Production Frontend | ✅ Ready (React + TypeScript) |
| Chat Assistant (Local LLM) | ✅ Ready (Ollama + SQLite) |
| Telemetry Visualizations | 🚧 Planned |

---

## 🛠️ Tech Stack

**Backend**
- Python 3.10+
- FastAPI - REST API framework
- LightGBM - Gradient boosting for ranking
- Optuna - Hyperparameter optimization
- FastF1 - F1 data API wrapper
- Pandas/NumPy - Data processing

**Frontend**
- React 18 + TypeScript
- Vite - Build tool
- Tailwind CSS - Styling
- TanStack React Query - Data fetching
- React Router - Routing
- Recharts - Visualizations

**Data & ML**
- Parquet - Efficient data storage
- Scikit-learn - ML utilities
- Joblib - Model serialization

---

## 📦 Current Data

**8 years of F1 data (2018-2025) already fetched:**

| Year | Races |
|------|-------|
| 2018 | 21 |
| 2019 | 21 |
| 2020 | 17 |
| 2021 | 22 |
| 2022 | 22 |
| 2023 | 22 |
| 2024 | 24 |
| 2025 | 24 |

**Total: 173 races, 3,978 driver-race rows**

---

## 🚀 SETUP

### Backend Setup

#### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

#### 2. Build Features (if not already done)

```bash
python -m backend.ml.build_features
```

Creates `data/fe/features.parquet` with all engineered features.

#### 3. Train Model

```bash
# Quick training (~1-2 min)
python -m backend.ml.trainer

# OR with Optuna hyperparameter tuning (~10-30 min, better results)
python -m backend.ml.trainer --tune --n-trials 50
```

#### 4. Start the API

```bash
python run_api.py
```

API will be available at `http://localhost:8000`

### Frontend Setup

#### 1. Install Node Dependencies

```bash
cd frontend
npm install
```

#### 2. Configure API URL (Optional)

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

#### 3. Start Development Server

```bash
npm run dev
```

Frontend will be available at `http://localhost:5173`

#### 4. Build for Production

```bash
npm run build
npm run preview
```

### Verify Everything Works

**Backend:**
```bash
# Test predictions via CLI
python -m backend.ml.predict --year 2024

# Or use API docs
# Open http://localhost:8000/docs
```

**Frontend:**
- Navigate to `http://localhost:5173`
- Check Dashboard for system status
- Use Predict page to generate predictions

---

## 🔄 2026 SEASON WORKFLOW

### 📅 Saturday (After Qualifying)

**Step 1: Fetch qualifying data**
```bash
python -m backend.ml.fetch_data --years 2026-2026
```

**Step 2: Build pre-race features**
```bash
python -m backend.ml.build_inference_rows --year 2026 --event "Bahrain Grand Prix"
```

**Step 3: Get predicted finishing order** 🏆
```bash
python -m backend.ml.predict --fe-path data/fe/inference/infer_2026_Bahrain_Grand_Prix.parquet
```

### 🏁 Sunday (After the Race)

**Update model with actual results:**
```bash
python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix"
```

This fetches race results → rebuilds features → retrains model.

**The model gets smarter with each race!** 📈

### Check Race Calendar

```bash
python -m backend.ml.update --list 2026
```

---

## 🛠️ CLI REFERENCE

### Fetch Data
```bash
# Fetch year range (USE 2018+ ONLY)
python -m backend.ml.fetch_data --years 2018-2025

# Fetch single year
python -m backend.ml.fetch_data --years 2024

# Fetch single race (for updates)
python -m backend.ml.fetch_data --race "2026_Bahrain Grand Prix"
```

### Build Features
```bash
# Build from all data
python -m backend.ml.build_features

# Build with year filter
python -m backend.ml.build_features --min-year 2020

# Custom output name
python -m backend.ml.build_features --output my_features.parquet
```

### Train Model
```bash
# Quick training (default params, ~1-2 min)
python -m backend.ml.trainer

# With Optuna hyperparameter tuning (~10-30 min, better results)
python -m backend.ml.trainer --tune

# Custom number of tuning trials
python -m backend.ml.trainer --tune --n-trials 100
```

### Run Predictions
```bash
# Predict all races in features file
python -m backend.ml.predict

# Filter by year
python -m backend.ml.predict --year 2024

# Filter by specific race
python -m backend.ml.predict --race-id "2024_Monaco Grand Prix"

# Use custom features file
python -m backend.ml.predict --fe-path data/fe/inference/infer_2026_Bahrain_Grand_Prix.parquet
```

### Rolling Update (Post-Race)
```bash
# Full update: fetch + features + retrain
python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix"

# Update without retraining
python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix" --no-retrain

# List races for a year
python -m backend.ml.update --list 2026
```

### Pre-Race Inference
```bash
# Build features for upcoming race (uses only pre-race data)
python -m backend.ml.build_inference_rows --year 2026 --event "Monaco Grand Prix"

# Include practice session data
python -m backend.ml.build_inference_rows --year 2026 --event "Monaco Grand Prix" --practice
```

---

## 🌐 API ENDPOINTS

Start the API:
```bash
python run_api.py
# Docs: http://127.0.0.1:8000/docs
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | GET | Get predictions (params: `year`, `race_id`) |
| `/update` | POST | Update after race (params: `year`, `race`) |
| `/races/{year}` | GET | List races for a year |
| `/status` | GET | System status (data years, model info) |

---

## 📁 PROJECT STRUCTURE

```
f1-project/
├── backend/
│   ├── api.py                    # FastAPI endpoints
│   └── ml/
│       ├── common.py             # Shared utilities & constants
│       ├── fetch_data.py         # Download from FastF1
│       ├── build_features.py     # Feature engineering
│       ├── build_inference_rows.py  # Pre-race features
│       ├── trainer.py            # Model training
│       ├── predict.py            # Run predictions
│       └── update.py             # Rolling update script
├── data/
│   ├── raw/                      # Raw parquet from FastF1
│   ├── fe/                       # Feature files
│   │   ├── features.parquet      # Main training features
│   │   └── inference/            # Pre-race prediction features
│   ├── processed/                # Intermediate data
│   └── preds/                    # Prediction CSV outputs
├── models/                       # Trained models (generated)
│   ├── ranker_lgb.txt            # LightGBM model
│   └── ranker_meta.joblib        # Model metadata
├── f1_cache/                     # FastF1 disk cache
├── frontend/                     # Simple web UI
├── requirements.txt
├── .gitignore
└── run_api.py                    # API launcher
```

---

## 📊 MODEL FEATURES (Pre-Race Only)

The model uses **only pre-race data** — no data leakage!

| Category | Features | When Known |
|----------|----------|------------|
| **Grid** | `grid_pos`, `quali_gap_s`, `grid_quali_diff` | After qualifying (Saturday) |
| **Form** | `driver_last3_avg_finish`, `team_last3_avg_finish` | Historical (past 3 races) |
| **Weather** | `mean_air_temp`, `mean_track_temp`, `is_wet_flag`, `wind_*` | Race morning forecast |
| **Tyres** | `start_compound`, `start_soft/medium/hard/inter/wet` | Formation lap |

### Current Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **NDCG@3** | ~0.86 | 86% podium prediction accuracy |
| **NDCG@10** | ~0.89 | 89% points positions accuracy |

> Post-race features (overtakes, pit losses, pos_change) are computed but **NOT used** for prediction — they would leak the answer!

---

## 📈 METRICS

| Metric | What It Measures |
|--------|-----------------|
| **NDCG@3** | Podium prediction quality |
| **NDCG@10** | Points positions prediction |
| **Top-3 Hit** | % of actual podium in predicted top 3 |
| **Spearman ρ** | Overall rank correlation |

---

## 🚧 TODO / FUTURE PLANS

### Phase 1: Core Improvements
- [x] ~~Hyperparameter tuning with Optuna~~ ✅ Done
- [ ] Add sprint race handling
- [ ] Add driver/team championship standings as features
- [ ] Track-specific features (street circuit, high downforce, etc.)
- [ ] Cross-validation for more robust evaluation

### Phase 2: Chat Assistant ✅
- [x] ~~Set up SQLite database for querying~~ ✅ Done
- [x] ~~Build NL→SQL pipeline with local LLM (Ollama)~~ ✅ Done
- [x] ~~Rule-based handlers for common queries~~ ✅ Done
- [x] ~~Natural language queries against dataset~~ ✅ Done
- See `backend/chatbot/README.md` for details

### Phase 3: Live Dashboard
- [ ] Real-time standings during race
- [ ] Prediction vs actual comparison
- [ ] Driver telemetry plots
- [ ] Tyre strategy visualization

### Phase 4: Advanced Models
- [ ] Experiment with neural ranking models
- [ ] Ensemble methods
- [ ] Lap-by-lap position prediction

---

## 🔧 TROUBLESHOOTING

### "Features file not found"
```bash
python -m backend.ml.build_features
```

### "Model not found"
```bash
python -m backend.ml.trainer
```

### "No data after filtering"
Check available races:
```bash
python -m backend.ml.update --list 2024
```

### FastF1 errors for old years (2015-2017)
**Don't use pre-2018 data.** FastF1 doesn't support it:
```
Cannot load laps, telemetry, weather, and message data because the relevant API is not supported for this session.
```

Use 2018-2025 instead:
```bash
python -m backend.ml.fetch_data --years 2018-2025
```

### Network timeout during fetch
The fetch can take a while. If it fails:
```bash
# Retry - FastF1 cache will skip already-downloaded sessions
python -m backend.ml.fetch_data --years 2018-2025
```

---

## 🙏 CREDITS

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry data
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
