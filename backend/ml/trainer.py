# backend/ml/trainer.py
"""
Train the LightGBM ranking model for F1 race predictions.
Uses LambdaRank objective optimized for NDCG.
"""
import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import FE_DIR, MODEL_DIR, META_COLS, TARGET_COLS

# Feature file path (unified)
FE_PATH = FE_DIR / "features.parquet"


def load_feature_table() -> pd.DataFrame:
    """
    Load and prepare the feature table for training.
    
    Returns:
        DataFrame with features converted to appropriate types
    """
    if not FE_PATH.exists():
        raise FileNotFoundError(
            f"Features not found at {FE_PATH}. "
            "Run: python -m backend.ml.build_features"
        )
    
    df = pd.read_parquet(FE_PATH)

    # Convert object columns to categorical (except metadata)
    all_meta = META_COLS | TARGET_COLS
    for c in df.columns:
        if df[c].dtype == "object" and c not in all_meta:
            df[c] = df[c].astype("category")

    return df


def make_groups(series: pd.Series) -> list[int]:
    """
    Build LightGBM group array (sizes per race, in row order).
    
    Args:
        series: Series containing group identifiers (e.g., race_id)
        
    Returns:
        List of group sizes preserving order of first appearance
    """
    sizes = series.value_counts()
    return [sizes[g] for g in series.drop_duplicates().tolist()]


def main():
    """Train the ranking model and save to disk."""
    df = load_feature_table()
    
    print(f"📊 Loaded {len(df):,} rows from {FE_PATH.name}")
    print(f"   Years: {df['event_year'].min()} - {df['event_year'].max()}")
    print(f"   Races: {df['race_id'].nunique()}")

    # Build feature list (everything except metadata/targets)
    features = [c for c in df.columns if c not in (META_COLS | TARGET_COLS)]

    # Ranking label: higher is better (inverse of finish position)
    max_pos = df["finish_pos"].max()
    df["rank_label"] = (max_pos - pd.to_numeric(df["finish_pos"], errors="coerce") + 1).fillna(0)
    df["rank_label"] = df["rank_label"].clip(lower=0)

    # Time-based split: most recent year = validation
    df = df.sort_values(["event_year", "event_name"]).reset_index(drop=True)
    
    if df["event_year"].nunique() >= 2:
        last_year = int(sorted(df["event_year"].unique())[-1])
        train = df[df["event_year"] < last_year].copy()
        valid = df[df["event_year"] == last_year].copy()
        print(f"\n🔀 Split: Train on <{last_year}, Validate on {last_year}")
    else:
        # Fallback: 30% races for validation
        hold_races = df["race_id"].drop_duplicates().sample(frac=0.3, random_state=42)
        valid = df[df["race_id"].isin(hold_races)].copy()
        train = df[~df["race_id"].isin(hold_races)].copy()
        print(f"\n🔀 Split: 70/30 random split (single year data)")

    # Categorical columns
    cat_cols = [c for c in features if str(train[c].dtype) == "category"]

    # Prepare datasets
    X_tr, y_tr, g_tr = train[features], train["rank_label"].values, make_groups(train["race_id"])
    X_va, y_va, g_va = valid[features], valid["rank_label"].values, make_groups(valid["race_id"])

    train_set = lgb.Dataset(
        X_tr, label=y_tr, group=g_tr,
        categorical_feature=cat_cols or "auto",
        free_raw_data=False
    )
    valid_set = lgb.Dataset(
        X_va, label=y_va, group=g_va,
        categorical_feature=cat_cols or "auto",
        reference=train_set,
        free_raw_data=False
    )

    # LambdaRank parameters
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 10],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_pre_filter=False,
        verbosity=-1,
    )

    print(f"\n🏋️ Training...")
    print(f"   Train: {len(train):,} rows ({len(g_tr)} races)")
    print(f"   Valid: {len(valid):,} rows ({len(g_va)} races)")
    print(f"   Features: {len(features)} ({len(cat_cols)} categorical)")

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )

    # Save model and metadata
    model_path = MODEL_DIR / "ranker_lgb.txt"
    model.save_model(str(model_path))
    
    meta_path = MODEL_DIR / "ranker_meta.joblib"
    joblib.dump({"features": features, "cat_cols": cat_cols}, meta_path)
    
    print(f"\n✅ Saved model → {model_path}")
    print(f"✅ Saved meta  → {meta_path}")

    # Compute validation metrics
    valid = valid.copy()
    valid["score"] = model.predict(valid[features])
    
    ndcgs = []
    for rid, g in valid.groupby("race_id", sort=False):
        rel = g["rank_label"].values
        order = np.argsort(-g["score"].values)
        
        # DCG@3
        gains = rel[order][:3]
        dcg = np.sum((2**gains - 1) / np.log2(np.arange(2, 2 + len(gains))))
        
        # Ideal DCG@3
        igains = np.sort(rel)[::-1][:3]
        idcg = np.sum((2**igains - 1) / np.log2(np.arange(2, 2 + len(igains))))
        
        if idcg > 0:
            ndcgs.append(dcg / idcg)
            
    if ndcgs:
        print(f"\n📈 Validation NDCG@3: {np.mean(ndcgs):.3f} (over {len(ndcgs)} races)")


if __name__ == "__main__":
    main()
