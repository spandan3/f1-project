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
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .ml.predict import predict, DEFAULT_FE, MODEL, META
from .ml.update import update_after_race
from .ml.fetch_data import get_available_races, fetch_single_race
from .ml.build_inference_rows import build_inference_from_cache

app = FastAPI(
    title="F1 Predictions API",
    description="Machine learning predictions for Formula 1 race outcomes",
    version="2.0.0",
)

# Manual CORS middleware to ensure it works
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "600",
            }
        )
    
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "F1 Predictions API", "version": "2.0.0"}


@app.get("/predict")
def api_predict(
    year: Optional[int] = Query(None, description="Filter by year (e.g., 2025)"),
    round: Optional[int] = Query(None, description="Race round number (e.g., 1)"),
    race_id: Optional[str] = Query(None, description="Exact race_id (e.g., 2025_Bahrain Grand Prix)"),
    fe_path: Optional[str] = Query(None, description="Custom features parquet path"),
):
    """
    Generate race predictions.
    
    Returns predicted finishing order with metrics if actual results are available.
    Can filter by year+round or by race_id.
    """
    try:
        # If round is provided, look up the race name
        race_name = None
        if round is not None and year is not None:
            races = get_available_races(year)
            race_match = next((r for r in races if r.get("round") == round), None)
            if not race_match:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Race round {round} not found for year {year}"
                )
            race_name = race_match['race_name']
            race_id = f"{year}_{race_name}"
        elif race_id:
            # Extract race name from race_id
            parts = race_id.split('_', 1)
            if len(parts) == 2:
                year = int(parts[0])
                race_name = parts[1]
        
        # Check if race exists in features.parquet (completed race)
        # If not, and it's 2026+, try to build inference features from qualifying
        use_inference = False
        if year and year >= 2026 and race_name and DEFAULT_FE.exists():
            try:
                df_check = pd.read_parquet(DEFAULT_FE)
                race_id_check = f"{year}_{race_name}"
                if race_id_check not in df_check['race_id'].values:
                    # Race not in completed data, try to build inference features
                    try:
                        print(f"📥 Building inference features for {year} {race_name}...")
                        inference_path = build_inference_from_cache(
                            event_year=year,
                            event_name=race_name,
                            n_hist_races=6,
                            use_practice=False
                        )
                        fe_path = str(inference_path)
                        use_inference = True
                    except Exception as e:
                        # If inference build fails, try with regular features anyway
                        print(f"⚠️ Could not build inference features: {e}")
                        pass
            except Exception:
                pass
        
        race_ids = [race_id] if race_id else None
        df, metrics = predict(race_ids=race_ids, year=year, fe_path=fe_path if use_inference else None, save_csv=False)
        
        # Extract race info from first row if available
        race_info = {}
        if len(df) > 0:
            race_info = {
                "year": int(df.iloc[0]["event_year"]),
                "round": round if round else None,
                "race_name": df.iloc[0]["event_name"],
            }
            # Try to get round from race list if not provided
            if not race_info["round"]:
                races = get_available_races(race_info["year"])
                race_match = next((r for r in races if r.get("race_name") == race_info["race_name"]), None)
                if race_match:
                    race_info["round"] = race_match.get("round")
        
        # Format predictions for frontend
        predictions = []
        for _, row in df.iterrows():
            predictions.append({
                "pred_pos": int(row["pred_rank"]),
                "driver": row["Driver"],
                "constructor": row["TeamName"],
                "grid_pos": int(row.get("grid_pos", 0)) if pd.notna(row.get("grid_pos")) else 0,
                "pred_score": float(row["score"]),
                "actual_pos": int(row["finish_pos"]) if "finish_pos" in df.columns and pd.notna(row.get("finish_pos", None)) else None,
                "finish_pos": int(row["finish_pos"]) if "finish_pos" in df.columns and pd.notna(row.get("finish_pos", None)) else None,
            })
        
        # Format metrics
        metrics_dict = None
        if metrics and len(metrics) > 0:
            m = metrics[0] if isinstance(metrics, list) else metrics
            metrics_dict = {
                "ndcg_3": float(m.get("ndcg@3", 0)),
                "ndcg_10": float(m.get("ndcg@10", 0)),
                "top3_hit": float(m.get("top3_hit", 0)),
                "spearman_rho": float(m.get("spearman_rho", 0)),
            }
        
        return {
            **race_info,
            "predictions": predictions,
            "metrics": metrics_dict,
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        # Handle missing columns gracefully (e.g., finish_pos for pre-race predictions)
        raise HTTPException(
            status_code=500, 
            detail=f"Missing required column: {str(e)}. This may indicate incomplete data."
        )
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ Error in /predict: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/races/{year}")
def list_races(year: int):
    """List available races for a given year."""
    races = get_available_races(year)
    return races  # Return array directly for frontend compatibility


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
        "status": "ready" if (model_exists and meta_exists and features_exist) else "setup_required",
        "model_exists": model_exists and meta_exists,
        "features_exist": features_exist,
        "available_years": [],
    }
    
    if features_exist:
        try:
            df = pd.read_parquet(DEFAULT_FE)
            available_years = sorted(df['event_year'].unique().tolist())
            info["available_years"] = [int(y) for y in available_years]
            info["data_years"] = f"{df['event_year'].min()}-{df['event_year'].max()}"
            info["total_races"] = df["race_id"].nunique()
            info["total_rows"] = len(df)
        except Exception:
            pass
    
    return info


@app.post("/prepare-race")
def prepare_race(
    year: int = Query(..., description="Race year (e.g., 2026)"),
    race: str = Query(..., description="GP name (e.g., 'Australian Grand Prix')"),
    build_features: bool = Query(True, description="Whether to build inference features after fetching"),
):
    """
    Prepare a race for prediction by fetching qualifying data and building features.
    
    This endpoint:
    1. Fetches qualifying (Q) and race (R) session data from FastF1
    2. Optionally builds inference features for pre-race prediction
    
    Use this before calling /predict for upcoming races.
    """
    try:
        # Step 1: Fetch data (only qualifying needed for pre-race predictions)
        print(f"📥 Fetching qualifying data for {year} {race}...")
        # For pre-race predictions, we only need qualifying (Q), not race (R)
        # This only fetches the specific race's qualifying data, not other races
        fetch_result = fetch_single_race(year, race, require_race=False)
        
        if not fetch_result.get("ok"):
            error_msg = fetch_result.get('error', 'Unknown error')
            # Provide more helpful error message
            if "not been loaded" in error_msg or "Session.load" in error_msg:
                error_msg = (
                    f"Qualifying data for {race} not available yet. "
                    f"This usually means: (1) Qualifying hasn't completed yet, "
                    f"(2) FastF1 hasn't received the data from F1 servers (wait 5-10 min), or "
                    f"(3) There's a network issue. "
                    f"Try again in a few minutes after qualifying completes."
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch qualifying data: {error_msg}"
            )
        
        result = {
            "ok": True,
            "year": year,
            "race": race,
            "data_fetched": True,
            "laps_added": fetch_result.get("laps_added", 0),
            "results_added": fetch_result.get("results_added", 0),
            "weather_added": fetch_result.get("weather_added", 0),
        }
        
        # Step 2: Build inference features if requested
        if build_features:
            try:
                print(f"🔧 Building inference features for {year} {race}...")
                inference_path = build_inference_from_cache(
                    event_year=year,
                    event_name=race,
                    n_hist_races=6,
                    use_practice=False
                )
                result["features_built"] = True
                result["features_path"] = str(inference_path)
            except Exception as e:
                result["features_built"] = False
                result["features_error"] = str(e)
                # Don't fail completely - data is fetched, features can be built later
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(
    question: str = Query(..., description="Natural language question about F1 data"),
    use_llm: bool = Query(True, description="Whether to use LLM fallback if rule-based fails")
):
    """
    Chatbot endpoint: Answer questions about F1 data.
    
    Uses rule-based handlers for common queries, falls back to LLM (Ollama) for arbitrary questions.
    All queries are executed as read-only SQL against the local database.
    """
    try:
        from .chatbot import ask_question
        response = ask_question(question, use_llm=use_llm)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
