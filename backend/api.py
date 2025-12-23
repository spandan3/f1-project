# backend/api.py

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .ml.predict import (
    DEFAULT_FE,
    MODEL,
    META,
    ndcg_at_k,
    spearman_rho_from_ranks,
    evaluate_group,
)

app = FastAPI(title="F1 Predictions API")

# Allow your frontend (e.g. localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_model(
    year: Optional[int] = None,
    race_id: Optional[str] = None,
    fe_path: Optional[str] = None,
):
    """Core logic: almost same as main() in predict.py but returns objects instead of printing/writing."""
    FE = Path(fe_path) if fe_path else DEFAULT_FE
    if not FE.exists():
        raise HTTPException(status_code=400, detail=f"Features file not found: {FE}")
    if not MODEL.exists() or not META.exists():
        raise HTTPException(status_code=500, detail="Model/meta not found. Train first.")

    df = pd.read_parquet(FE)
    meta = joblib.load(META)
    features: list[str] = meta["features"]
    cat_cols: list[str] = meta.get("cat_cols", [])

    # Filter
    if year is not None:
        df = df[df["event_year"] == year].copy()
    if race_id:
        df = df[df["race_id"] == race_id].copy()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No rows after filtering. Check year/race_id.",
        )

    # Ensure features exist
    missing = [c for c in features if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    for c in cat_cols:
        if c in df.columns and df[c].dtype.name != "category":
            df[c] = df[c].astype("category")

    # Predict
    model = lgb.Booster(model_file=str(MODEL))
    X = df[features]
    df["score"] = model.predict(X)
    df = df.sort_values(["race_id", "score"], ascending=[True, False])
    df["pred_rank"] = df.groupby("race_id")["score"].rank(
        ascending=False, method="first"
    )

    # Columns for frontend
    cols = [
        "race_id",
        "event_year",
        "event_name",
        "Driver",
        "TeamName",
        "grid_pos",
        "finish_pos",
        "pred_rank",
        "score",
    ]
    cols = [c for c in cols if c in df.columns]

    # Metrics (if finish_pos available)
    metrics_summary = None
    if "finish_pos" in df.columns and not df["finish_pos"].isna().all():
        metrics = []
        for _, g in df.groupby("race_id", sort=False):
            if g["finish_pos"].isna().any():
                continue
            metrics.append(evaluate_group(g))

        if metrics:
            met_df = pd.DataFrame(metrics).sort_values(["event_year", "race_id"])
            summary = met_df[
                ["ndcg@3", "ndcg@10", "top3_hit", "spearman_rho"]
            ].mean(numeric_only=True)
            metrics_summary = {
                "ndcg@3": float(summary["ndcg@3"]),
                "ndcg@10": float(summary["ndcg@10"]),
                "top3_hit": float(summary["top3_hit"]),
                "spearman_rho": float(summary["spearman_rho"]),
            }

    return df[cols], metrics_summary


@app.get("/predict")
def predict(
    year: Optional[int] = Query(None, description="Filter by event_year (e.g., 2025)."),
    race_id: Optional[str] = Query(
        None,
        description="Exact race_id (e.g., 2025_Bahrain Grand Prix). If omitted, scores all races for that year.",
    ),
    fe_path: Optional[str] = Query(
        None,
        description="Optional custom features parquet path.",
    ),
):
    """HTTP endpoint for predictions."""
    df, metrics = _run_model(year=year, race_id=race_id, fe_path=fe_path)

    predictions = df.to_dict(orient="records")
    return {
        "predictions": predictions,
        "metrics": metrics,  # may be null if finish_pos not available
    }

# --- DATA REFRESH ENDPOINT (manual trigger) ---

from backend.ml.build_2025_dataset import fetch_2025_raw


@app.post("/refresh")
def refresh(year: int = 2025):
    """
    Rebuild / update feature dataset for a season.
    This will:
    - pull latest FastF1 data
    - update quali / sprint features if available
    - ensure all races exist in the FE parquet
    """
    fetch_2025_raw()
    return {
        "ok": True,
        "message": "Feature dataset refreshed",
        "year": year,
    }

