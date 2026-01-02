# 🏎️ F1 Race Prediction System

A comprehensive machine learning system that predicts Formula 1 race finishing order with an interactive web interface and AI-powered database assistant. Built for **rolling predictions** throughout the 2026 season.

**Features:**
- 🎯 **Race Outcome Prediction** - LightGBM ranking model for podium and finishing order predictions
- 📊 **Interactive Dashboard** - Real-time predictions, analysis, and visualizations
- 🤖 **AI Database Assistant** - Natural language queries powered by Groq/Ollama
- 📈 **Performance Analytics** - Driver/team trends, position changes, and strategy insights
- 🔄 **Rolling Updates** - Model improves after each race with new data

---

## 📸 Demo

### Landing Page
*Insert landing page screenshot here*

The main entry point where users select a season and race to generate predictions.

---

### Race Dashboard
*Insert dashboard screenshot here*

**Features:**
- Predicted finishing order with confidence scores
- Podium highlighting (P1-P3)
- Predicted vs. actual comparison (when available)
- Interactive sorting and filtering
- Performance metrics (NDCG@3, NDCG@10, Top-3 Hit, Spearman ρ)
- Visual charts: Grid position vs. predicted finish, position deltas

---

### Prediction Table
*Insert prediction table screenshot here*

Detailed table showing:
- Predicted position
- Driver and team
- Grid position
- Predicted score/confidence
- Actual position (if race completed)
- Position delta (+/-)

---

### Upcoming Races
*Insert upcoming races page screenshot here*

Dedicated page for future race predictions with automatic detection of next available race.

---

### AI Database Assistant
*Insert chatbot interface screenshot here*

**Natural Language Query Interface:**
- Ask questions in plain English
- Powered by Groq (Llama 3.1 70B) or Ollama (local)
- Query 8 years of F1 data (2018-2025)
- View generated SQL queries
- Interactive data tables for results

**Example Questions:**
- "Who won the 2023 championship?"
- "Who finished 2nd in Monaco 2024?"
- "Who had the fastest lap in Bahrain 2023?"
- "How many podiums did Verstappen get in 2024?"

---

### Performance Metrics
*Insert metrics display screenshot here*

Real-time model performance metrics including NDCG scores, hit rates, and correlation coefficients.

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

**Chatbot & AI**
- Groq API - Cloud LLM (Llama 3.1 70B)
- Ollama - Local LLM support (DeepSeek Coder 6.7B)
- SQLite - Database for natural language queries
- Python-dotenv - Environment variable management

---

## 📦 Current Data

**8 years of F1 data (2018-2025) already fetched:**

**Total: 173 races, 3,978 driver-race rows**


---


## 🌐 API ENDPOINTS



| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | GET | Get predictions (params: `year`, `round`) |
| `/update` | POST | Update after race (params: `year`, `round`, `retrain`) |
| `/races/{year}` | GET | List races for a year |
| `/status` | GET | System status (data years, model info) |
| `/chat` | POST | AI database assistant (params: `question`, `use_llm`) |

---

## 📊 MODEL FEATURES (Pre-Race Only)

The model uses **only pre-race data**

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


---

## 📈 METRICS

| Metric | What It Measures |
|--------|-----------------|
| **NDCG@3** | Podium prediction quality |
| **NDCG@10** | Points positions prediction |
| **Top-3 Hit** | % of actual podium in predicted top 3 |
| **Spearman ρ** | Overall rank correlation |

---

## 🤖 AI Database Assistant

The chatbot allows natural language queries against the F1 database:

**Usage:**
- Ask questions in plain English
- View generated SQL queries
- Get instant answers from 8 years of F1 data

**Supported Queries:**
- Championship standings ("Who won the 2023 championship?")
- Race positions ("Who finished 2nd in Monaco 2024?")
- Lap times ("Who had the fastest lap in Bahrain 2023?")
- Statistics ("How many podiums did Verstappen get in 2024?")

See `backend/chatbot/README.md` for detailed documentation.

---

## 🙏 CREDITS

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry data
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Groq](https://groq.com/) - AI inference platform (free tier)
- [Ollama](https://ollama.ai/) - Local LLM runtime

