# 🏎️ F1 Race Prediction System

A machine learning system that predicts Formula 1 race outcomes. Designed for **rolling predictions** throughout the 2026 season - the model improves as more races are added.

## 🎯 Features

### ✅ Race Outcome Prediction
- **LightGBM LambdaRank model** predicts finishing order
- Uses: qualifying results, driver/team form, weather, tyre strategy, pit stops
- Trained on 10 years of historical data (2015-2025)

### ✅ Rolling Updates
- Add new race data after each GP
- Automatic feature rebuild + model retrain
- Model improves throughout the season

### ✅ REST API
- `/predict` - Get race predictions
- `/update` - Add race and retrain (post-race)
- `/status` - System status
- `/races/{year}` - List races for a year

### 🚧 Planned
- Interactive Chat Assistant (RAG)
- Live Dashboard
- Driver telemetry visualizations

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Fetch 10 Years of Data (One-Time)

```bash
# This takes a while - downloads 2015-2025 data from FastF1
python -m backend.ml.fetch_data --years 2015-2025
```

### 3. Build Features

```bash
python -m backend.ml.build_features
```

### 4. Train Model

```bash
python -m backend.ml.trainer
```

### 5. Run Predictions

```bash
# Predict all races
python -m backend.ml.predict

# Predict specific year
python -m backend.ml.predict --year 2025

# Predict specific race
python -m backend.ml.predict --race-id "2025_Bahrain Grand Prix"
```

### 6. Start API

```bash
python run_api.py
# API: http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
```

---

## 🔄 2026 Rolling Workflow

After each 2026 race completes:

```bash
# One command: fetch + rebuild features + retrain
python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix"
```

Or via API:
```bash
curl -X POST "http://127.0.0.1:8000/update?year=2026&race=Bahrain%20Grand%20Prix"
```

The model will improve as the season progresses!

---

## 📁 Project Structure

```
f1-project/
├── backend/
│   ├── api.py                 # FastAPI endpoints
│   └── ml/
│       ├── common.py          # Shared utilities
│       ├── fetch_data.py      # Download data from FastF1
│       ├── build_features.py  # Feature engineering
│       ├── trainer.py         # Model training
│       ├── predict.py         # Run predictions
│       ├── update.py          # Rolling update (fetch→build→train)
│       └── build_inference_rows.py  # Pre-race predictions
├── data/
│   ├── raw/                   # Raw parquet from FastF1
│   ├── fe/                    # Feature files
│   │   └── features.parquet   # Main feature dataset
│   └── preds/                 # Prediction outputs
├── models/                    # Trained models (generated)
├── frontend/                  # Simple web UI
├── requirements.txt
└── run_api.py
```

---

## 📊 Model Features

| Category | Features |
|----------|----------|
| **Grid** | `grid_pos`, `quali_gap_s`, `grid_quali_diff` |
| **Form** | `driver_last3_avg_finish`, `team_last3_avg_finish` |
| **Race** | `pos_change`, `race_total_overtakes`, `driver_net_passes` |
| **Weather** | `mean_air_temp`, `mean_track_temp`, `is_wet_flag` |
| **Strategy** | `start_compound`, `pit_loss_total_s` |

---

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@3** | Ranking quality for podium |
| **NDCG@10** | Ranking quality for points |
| **Top-3 Hit** | Podium prediction accuracy |
| **Spearman ρ** | Overall rank correlation |

---

## 🛠️ CLI Reference

```bash
# Fetch data
python -m backend.ml.fetch_data --years 2015-2025
python -m backend.ml.fetch_data --race "2026_Bahrain Grand Prix"

# Build features
python -m backend.ml.build_features
python -m backend.ml.build_features --min-year 2020

# Train
python -m backend.ml.trainer

# Predict
python -m backend.ml.predict --year 2025
python -m backend.ml.predict --race-id "2025_Monaco Grand Prix"

# Rolling update (after race)
python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix"
python -m backend.ml.update --list 2026  # Show race calendar
```

---

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry data
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting
