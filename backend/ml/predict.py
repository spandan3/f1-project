# backend/ml/predict.py
"""
Run predictions using trained LightGBM ranking model.
Supports filtering by year and race_id, outputs predictions and metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import BASE_DIR, FE_DIR, MODEL_DIR

# Default paths
DEFAULT_FE = FE_DIR / "features.parquet"
MODEL = MODEL_DIR / "ranker_lgb.txt"
META = MODEL_DIR / "ranker_meta.joblib"
OUT_DIR = BASE_DIR / "data" / "preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ndcg_at_k(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    gains = rel[order][:k]
    dcg = np.sum((2**gains - 1) / np.log2(np.arange(2, 2 + len(gains))))
    igains = np.sort(rel)[::-1][:k]
    idcg = np.sum((2**igains - 1) / np.log2(np.arange(2, 2 + len(igains))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def spearman_rho_from_ranks(pred_rank: np.ndarray, true_rank: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(pred_rank) < 2 or len(true_rank) < 2:
        return np.nan
    a = np.asarray(pred_rank, dtype=float)
    b = np.asarray(true_rank, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.clip((a * b).mean(), -1.0, 1.0))


def evaluate_group(g: pd.DataFrame) -> dict:
    """Evaluate prediction metrics for a single race."""
    # Convert finish_pos to relevance: lower position = higher relevance
    # Same formula as trainer.py: max_pos - finish_pos + 1
    finish_pos = pd.to_numeric(g["finish_pos"], errors="coerce").values
    max_pos = max(finish_pos[~pd.isna(finish_pos)]) if len(finish_pos[~pd.isna(finish_pos)]) > 0 else 20
    rel = (max_pos - finish_pos + 1)
    rel = np.nan_to_num(rel, nan=0.0)  # Handle NaN
    
    scores = g["score"].values

    true_rank = g["finish_pos"].rank(method="min").values
    pred_rank = g["pred_rank"].values

    ndcg3 = ndcg_at_k(scores, rel, k=3)
    ndcg10 = ndcg_at_k(scores, rel, k=10)

    # Top-3 hit rate
    top3_pred_idx = np.argsort(-scores)[:3]
    top3_pred_drivers = set(g.iloc[top3_pred_idx]["Driver"])
    top3_true_idx = np.argsort(g["finish_pos"].values)[:3]
    top3_true_drivers = set(g.iloc[top3_true_idx]["Driver"])
    top3_hit = len(top3_pred_drivers & top3_true_drivers) / 3.0

    rho = spearman_rho_from_ranks(pred_rank, true_rank)

    return {
        "race_id": g["race_id"].iloc[0],
        "event_year": int(g["event_year"].iloc[0]),
        "event_name": g["event_name"].iloc[0],
        "ndcg@3": ndcg3,
        "ndcg@10": ndcg10,
        "top3_hit": top3_hit,
        "spearman_rho": rho,
    }


def predict(
    race_ids: list[str] | None = None,
    year: int | None = None,
    fe_path: str | None = None,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Run predictions and return results.
    
    Args:
        race_ids: Optional list of race_ids to filter
        year: Optional year to filter
        fe_path: Optional custom features file path
        save_csv: Whether to save CSV output files
        
    Returns:
        Tuple of (predictions DataFrame, metrics dict or None)
    """
    # Load features + model
    FE = Path(fe_path) if fe_path else DEFAULT_FE
    if not FE.exists():
        raise FileNotFoundError(
            f"Features file not found: {FE}. "
            "Run: python -m backend.ml.build_features"
        )
    if not MODEL.exists() or not META.exists():
        raise FileNotFoundError(
            "Model not found. Run: python -m backend.ml.trainer"
        )

    df = pd.read_parquet(FE)
    meta = joblib.load(META)
    features: list[str] = meta["features"]
    cat_cols: list[str] = meta.get("cat_cols", [])

    # Apply filters
    scope = []
    if year is not None:
        df = df[df["event_year"] == year].copy()
        scope.append(str(year))
    if race_ids:
        df = df[df["race_id"].isin(race_ids)].copy()
        scope.extend(race_ids)

    if df.empty:
        raise ValueError(f"No data after filtering. Check --year/--race-id.")

    # Ensure all expected features exist
    missing = [c for c in features if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    # Cast categories
    for c in cat_cols:
        if c in df.columns and df[c].dtype.name != "category":
            df[c] = df[c].astype("category")

    # Run prediction
    model = lgb.Booster(model_file=str(MODEL))
    X = df[features]
    df["score"] = model.predict(X)
    df = df.sort_values(["race_id", "score"], ascending=[True, False])
    df["pred_rank"] = df.groupby("race_id")["score"].rank(ascending=False, method="first")

    # Output columns (finish_pos may not exist for pre-race predictions)
    cols = ["race_id", "event_year", "event_name", "Driver", "TeamName", 
            "grid_pos", "pred_rank", "score"]
    # Only add finish_pos if it exists (for completed races)
    if "finish_pos" in df.columns:
        cols.append("finish_pos")
    cols = [c for c in cols if c in df.columns]
    
    result_df = df[cols].copy()

    # Compute metrics if we have actual results
    metrics_summary = None
    if "finish_pos" in df.columns and not df["finish_pos"].isna().all():
        metrics = []
        for _, g in df.groupby("race_id", sort=False):
            if g["finish_pos"].isna().any():
                continue
            metrics.append(evaluate_group(g))

        if metrics:
            met_df = pd.DataFrame(metrics).sort_values(["event_year", "race_id"])
            summary = met_df[["ndcg@3", "ndcg@10", "top3_hit", "spearman_rho"]].mean()
            metrics_summary = {
                "ndcg@3": float(summary["ndcg@3"]),
                "ndcg@10": float(summary["ndcg@10"]),
                "top3_hit": float(summary["top3_hit"]),
                "spearman_rho": float(summary["spearman_rho"]),
            }
            
            if save_csv:
                out_metrics = OUT_DIR / (
                    "metrics_all.csv" if not scope 
                    else f"metrics_{'_'.join(s.replace(' ', '_') for s in scope)}.csv"
                )
                met_df.to_csv(out_metrics, index=False)
                print(f"📊 Saved metrics → {out_metrics}")

    # Save predictions
    if save_csv:
        out_pred = OUT_DIR / (
            "predictions_all.csv" if not scope 
            else f"predictions_{'_'.join(s.replace(' ', '_') for s in scope)}.csv"
        )
        result_df.to_csv(out_pred, index=False)
        print(f"✅ Saved predictions → {out_pred}")

    return result_df, metrics_summary


def main(race_ids: list[str] | None, year: int | None, fe_path: str | None):
    """CLI entry point."""
    result_df, metrics = predict(race_ids, year, fe_path, save_csv=True)

    # Print table
    print("\n=== Predictions ===")
    with pd.option_context('display.max_rows', 50, 'display.max_columns', None):
        print(result_df.to_string(index=False))

    # Print metrics summary
    if metrics:
        print(
            f"\n📈 Overall — NDCG@3: {metrics['ndcg@3']:.3f} | "
            f"NDCG@10: {metrics['ndcg@10']:.3f} | "
            f"Top-3 hit: {metrics['top3_hit']:.3f} | "
            f"Spearman ρ: {metrics['spearman_rho']:.3f}"
        )
    else:
        print("\nℹ️ No finish_pos available; metrics not computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run F1 race predictions")
    parser.add_argument(
        "--race-id", nargs="*", default=None,
        help="One or more race_ids (e.g., 2025_Bahrain Grand Prix)"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Filter by year (e.g., 2025)"
    )
    parser.add_argument(
        "--fe-path", type=str, default=None,
        help="Path to features parquet file"
    )
    args = parser.parse_args()
    main(args.race_id, args.year, args.fe_path)
