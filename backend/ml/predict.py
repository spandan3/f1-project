# backend/ml/predict.py
import argparse
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FE = BASE_DIR / "data" / "fe" / "2025" / "standings_2025.parquet"
MODEL     = BASE_DIR / "models" / "ranker_lgb.txt"
META      = BASE_DIR / "models" / "ranker_meta.joblib"
OUT_DIR   = BASE_DIR / "data" / "preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ndcg_at_k(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    gains = rel[order][:k]
    dcg = np.sum((2**gains - 1) / np.log2(np.arange(2, 2 + len(gains))))
    igains = np.sort(rel)[::-1][:k]
    idcg = np.sum((2**igains - 1) / np.log2(np.arange(2, 2 + len(igains))))
    return float(dcg / idcg) if idcg > 0 else 0.0

def spearman_rho_from_ranks(pred_rank: np.ndarray, true_rank: np.ndarray) -> float:
    # Spearman = Pearson on ranks
    if len(pred_rank) < 2 or len(true_rank) < 2:
        return np.nan
    a = np.asarray(pred_rank, dtype=float)
    b = np.asarray(true_rank, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.clip((a * b).mean(), -1.0, 1.0))

def evaluate_group(g: pd.DataFrame) -> dict:
    # Relevance for ranking metrics: higher = better; use inverse finish position
    rel = -pd.to_numeric(g["finish_pos"], errors="coerce").values
    scores = g["score"].values

    # True rank from finish_pos (1 = winner)
    true_rank = g["finish_pos"].rank(method="min").values
    pred_rank = g["pred_rank"].values

    ndcg3  = ndcg_at_k(scores, rel, k=3)
    ndcg10 = ndcg_at_k(scores, rel, k=10)

    # Top-3 hit rate: fraction of actual podium found in predicted top 3
    top3_pred_idx = np.argsort(-scores)[:3]
    top3_pred_drivers = set(g.iloc[top3_pred_idx]["Driver"])
    top3_true_idx = np.argsort(g["finish_pos"].values)[:3]  # smallest finish_pos = best
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

def main(race_ids: list[str] | None, year: int | None, fe_path: str | None):
    # Load features + model
    FE = Path(fe_path) if fe_path else DEFAULT_FE
    if not FE.exists():
        raise FileNotFoundError(f"Features file not found: {FE}")
    if not MODEL.exists() or not META.exists():
        raise FileNotFoundError("Model/meta not found. Train first.")

    df = pd.read_parquet(FE)
    meta = joblib.load(META)
    features: list[str] = meta["features"]
    cat_cols: list[str] = meta.get("cat_cols", [])

    # Optional filters
    scope = []
    if year is not None:
        df = df[df["event_year"] == year].copy()
        scope.append(str(year))
    if race_ids:
        df = df[df["race_id"].isin(race_ids)].copy()
        scope.extend(race_ids)

    if df.empty:
        # help user with available options
        avail = df["race_id"].unique().tolist() if "race_id" in df.columns else []
        raise ValueError(f"No rows after filtering. Check --year/--race-id. Available race_ids example: {avail[:20]}")

    # Ensure all expected features exist; fill missing numeric as 0.0
    missing = [c for c in features if c not in df.columns]
    for c in missing:
        df[c] = 0.0  # safe default for numeric engineered features

    # Cast categories for inference
    for c in cat_cols:
        if c in df.columns and df[c].dtype.name != "category":
            df[c] = df[c].astype("category")

    # Predict
    model = lgb.Booster(model_file=str(MODEL))
    X = df[features]
    df["score"] = model.predict(X)
    df = df.sort_values(["race_id", "score"], ascending=[True, False])
    df["pred_rank"] = df.groupby("race_id")["score"].rank(ascending=False, method="first")

    # Write predictions
    out_pred = OUT_DIR / (
        "predictions_all.csv" if not scope else f"predictions_{'_'.join(s.replace(' ', '_') for s in scope)}.csv"
    )
    cols = ["race_id","event_year","event_name","Driver","TeamName","grid_pos","finish_pos","pred_rank","score"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_pred, index=False)
    print(f"✅ Wrote {out_pred}")

    # Print table to console like before
    print("\n=== Predictions Table ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[cols].to_string(index=False))


    # If no finish_pos (upcoming races), we can’t compute metrics
    if "finish_pos" not in df.columns or df["finish_pos"].isna().all():
        print("ℹ️ No finish_pos available; skipping metrics.")
        return

    # Per‑race metrics
    metrics = []
    for _, g in df.groupby("race_id", sort=False):
        # skip races missing finish_pos
        if g["finish_pos"].isna().any():
            continue
        metrics.append(evaluate_group(g))

    if metrics:
        met_df = pd.DataFrame(metrics).sort_values(["event_year", "race_id"])
        out_metrics = OUT_DIR / (
            "metrics_all.csv" if not scope else f"metrics_{'_'.join(s.replace(' ', '_') for s in scope)}.csv"
        )
        met_df.to_csv(out_metrics, index=False)
        print(f"📊 Saved per‑race metrics → {out_metrics}")

        # Overall summary
        summary = met_df[["ndcg@3", "ndcg@10", "top3_hit", "spearman_rho"]].mean(numeric_only=True)
        print(
            f"📈 Overall — NDCG@3: {summary['ndcg@3']:.3f} | "
            f"NDCG@10: {summary['ndcg@10']:.3f} | "
            f"Top‑3 hit: {summary['top3_hit']:.3f} | "
            f"Spearman ρ: {summary['spearman_rho']:.3f}"
        )
    else:
        print("ℹ️ No complete races with finish_pos to score.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-id", nargs="*", default=None,
                        help="One or more race_ids (e.g., 2025_Bahrain Grand Prix). Omit to score all races in the file.")
    parser.add_argument("--year", type=int, default=None, help="Filter by event_year (e.g., 2025).")
    parser.add_argument("--fe-path", type=str, default=None,
                        help="Path to features parquet (e.g., data/fe/2025/standings_2025.parquet).")
    args = parser.parse_args()
    main(args.race_id, args.year, args.fe_path)
