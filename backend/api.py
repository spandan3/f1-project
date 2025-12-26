# backend/api.py
"""
FastAPI backend for F1 race predictions.
"""
import os
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .ml.predict import predict, DEFAULT_FE, MODEL, META
from .ml.update import update_after_race
from .ml.fetch_data import get_available_races

app = FastAPI(
    title="F1 Predictions API",
    description="Machine learning predictions for Formula 1 race outcomes",
    version="2.0.0",
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "F1 Predictions API", "version": "2.0.0"}


@app.get("/predict")
def api_predict(
    year: Optional[int] = Query(None, description="Filter by year (e.g., 2025)"),
    race_id: Optional[str] = Query(None, description="Exact race_id (e.g., 2025_Bahrain Grand Prix)"),
    fe_path: Optional[str] = Query(None, description="Custom features parquet path"),
):
    """
    Generate race predictions.
    
    Returns predicted finishing order with metrics if actual results are available.
    """
    try:
        race_ids = [race_id] if race_id else None
        df, metrics = predict(race_ids=race_ids, year=year, fe_path=fe_path, save_csv=False)
        
        return {
            "predictions": df.to_dict(orient="records"),
            "metrics": metrics,
            "count": len(df),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/races/{year}")
def list_races(year: int):
    """List available races for a given year."""
    races = get_available_races(year)
    return {"year": year, "races": races, "count": len(races)}


@app.post("/update")
def api_update(
    year: int = Query(..., description="Race year (e.g., 2026)"),
    race: str = Query(..., description="GP name (e.g., 'Bahrain Grand Prix')"),
    retrain: bool = Query(True, description="Whether to retrain the model"),
):
    """
    Update after a race completes.
    
    Fetches new race data, rebuilds features, and optionally retrains the model.
    """
    try:
        result = update_after_race(year, race, retrain=retrain)
        if not result["ok"]:
            raise HTTPException(
                status_code=500,
                detail=f"Update failed at step '{result.get('step')}': {result.get('error')}"
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    """Get current system status."""
    model_exists = MODEL.exists()
    meta_exists = META.exists()
    features_exist = DEFAULT_FE.exists()
    
    info = {
        "model_trained": model_exists and meta_exists,
        "features_available": features_exist,
    }
    
    if features_exist:
        try:
            df = pd.read_parquet(DEFAULT_FE)
            info["data_years"] = f"{df['event_year'].min()}-{df['event_year'].max()}"
            info["total_races"] = df["race_id"].nunique()
            info["total_rows"] = len(df)
        except Exception:
            pass
    
    return info
