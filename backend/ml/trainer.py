import os, joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
FE_PATH  = BASE_DIR / "data" / "fe" / "standings_train.parquet"
MODEL_DIR= BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_feature_table() -> pd.DataFrame:
    df = pd.read_parquet(FE_PATH)

    # DO NOT feed identifiers as features
    meta_cols = {"race_id","event_year","event_name","Driver","DriverNumber","TeamName","finish_pos","scored_points"}

    # Detect categorical candidates (strings that should be categories)
    # Example: start_compound; keep driver/team names as metadata, not features
    for c in df.columns:
        if df[c].dtype == "object" and c not in meta_cols:
            df[c] = df[c].astype("category")

    return df

def make_groups(series: pd.Series) -> list[int]:
    # Build LightGBM group array (sizes per race, in row order)
    sizes = series.value_counts()
    # Preserve order of first appearance per group
    return [sizes[g] for g in series.drop_duplicates().tolist()]

def main():
    df = load_feature_table()

    # Build feature list automatically (everything except metadata/targets)
    meta_cols = {"race_id","event_year","event_name","Driver","DriverNumber","TeamName"}
    target_cols = {"finish_pos","scored_points"}
    features = [c for c in df.columns if c not in (meta_cols | target_cols)]

    # Ranking label: negative finishing position (higher is better)
    max_pos = df["finish_pos"].max()
    df["rank_label"] = (max_pos - pd.to_numeric(df["finish_pos"], errors="coerce") + 1).fillna(0)

    # Optional: cap negatives (in case of DNFs or NaNs)
    df["rank_label"] = df["rank_label"].clip(lower=0)

    # Time-based split: last season = validation (fallback to 30% races if only 1 year)
    df = df.sort_values(["event_year","event_name"]).reset_index(drop=True)
    if df["event_year"].nunique() >= 2:
        last_year = int(sorted(df["event_year"].unique())[-1])
        train = df[df["event_year"] < last_year].copy()
        valid = df[df["event_year"] == last_year].copy()
    else:
        hold_races = df["race_id"].drop_duplicates().sample(frac=0.3, random_state=42)
        valid = df[df["race_id"].isin(hold_races)].copy()
        train = df[~df["race_id"].isin(hold_races)].copy()

    # Categorical columns = pandas categorical dtype
    cat_cols = [c for c in features if str(train[c].dtype) == "category"]

    # Prepare datasets
    X_tr, y_tr, g_tr = train[features], train["rank_label"].values, make_groups(train["race_id"])
    X_va, y_va, g_va = valid[features], valid["rank_label"].values, make_groups(valid["race_id"])

    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, categorical_feature=cat_cols or "auto", free_raw_data=False)
    valid_set = lgb.Dataset(X_va, label=y_va, group=g_va, categorical_feature=cat_cols or "auto", reference=train_set, free_raw_data=False)

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

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )

    # Save
    model_path = MODEL_DIR / "ranker_lgb.txt"
    model.save_model(model_path)
    joblib.dump(
        {"features": features, "cat_cols": cat_cols},
        MODEL_DIR / "ranker_meta.joblib"
    )
    print(f"✅ Saved model → {model_path}")
    print(f"✅ Saved meta  → {MODEL_DIR / 'ranker_meta.joblib'}")

    # Quick per‑race NDCG@3 (approx)
    valid = valid.copy()
    valid["score"] = model.predict(valid[features])
    ndcgs = []
    for rid, g in valid.groupby("race_id", sort=False):
        rel = g["rank_label"].values
        order = np.argsort(-g["score"].values)  # best first
        # DCG@3
        gains = rel[order][:3]
        dcg = np.sum((2**gains - 1) / np.log2(np.arange(2, 2+len(gains))))
        # Ideal DCG@3
        igains = np.sort(rel)[::-1][:3]
        idcg = np.sum((2**igains - 1) / np.log2(np.arange(2, 2+len(igains))))
        if idcg > 0:
            ndcgs.append(dcg/idcg)
    if ndcgs:
        print(f"📈 Mean NDCG@3 on validation: {np.mean(ndcgs):.3f} over {len(ndcgs)} races")

if __name__ == "__main__":
    main()
