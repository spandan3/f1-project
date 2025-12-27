# backend/ml/trainer.py
"""
Train the LightGBM ranking model for F1 race predictions.
Supports both quick training and Optuna hyperparameter optimization.
"""
import argparse
import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import FE_DIR, MODEL_DIR, META_COLS, TARGET_COLS, PRE_RACE_FEATURES

# Feature file path
FE_PATH = FE_DIR / "features.parquet"


def load_feature_table() -> pd.DataFrame:
    """Load and prepare the feature table for training."""
    if not FE_PATH.exists():
        raise FileNotFoundError(
            f"Features not found at {FE_PATH}. "
            "Run: python -m backend.ml.build_features"
        )
    
    df = pd.read_parquet(FE_PATH)

    # Convert object columns to categorical
    all_meta = META_COLS | TARGET_COLS
    for c in df.columns:
        if df[c].dtype == "object" and c not in all_meta:
            df[c] = df[c].astype("category")

    return df


def make_groups(series: pd.Series) -> list[int]:
    """Build LightGBM group array (sizes per race)."""
    sizes = series.value_counts()
    return [sizes[g] for g in series.drop_duplicates().tolist()]


def prepare_data(df: pd.DataFrame):
    """Prepare train/validation split and datasets."""
    # IMPORTANT: Only use PRE-RACE features to avoid data leakage!
    # Post-race features (pos_change, overtakes, pit_loss) would leak the answer
    features = [c for c in PRE_RACE_FEATURES if c in df.columns]

    # Ranking label: higher is better
    max_pos = df["finish_pos"].max()
    df["rank_label"] = (max_pos - pd.to_numeric(df["finish_pos"], errors="coerce") + 1).fillna(0)
    df["rank_label"] = df["rank_label"].clip(lower=0)

    # Time-based split
    df = df.sort_values(["event_year", "event_name"]).reset_index(drop=True)
    
    if df["event_year"].nunique() >= 2:
        last_year = int(sorted(df["event_year"].unique())[-1])
        train = df[df["event_year"] < last_year].copy()
        valid = df[df["event_year"] == last_year].copy()
    else:
        hold_races = df["race_id"].drop_duplicates().sample(frac=0.3, random_state=42)
        valid = df[df["race_id"].isin(hold_races)].copy()
        train = df[~df["race_id"].isin(hold_races)].copy()

    cat_cols = [c for c in features if str(train[c].dtype) == "category"]

    return train, valid, features, cat_cols


def train_with_params(train, valid, features, cat_cols, params: dict):
    """Train model with specific parameters."""
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

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],  # Silent
    )

    return model


def compute_ndcg(model, valid, features) -> float:
    """Compute mean NDCG@3 on validation set."""
    valid = valid.copy()
    valid["score"] = model.predict(valid[features])
    
    ndcgs = []
    for _, g in valid.groupby("race_id", sort=False):
        rel = g["rank_label"].values
        order = np.argsort(-g["score"].values)
        
        gains = rel[order][:3]
        dcg = np.sum((2**gains - 1) / np.log2(np.arange(2, 2 + len(gains))))
        
        igains = np.sort(rel)[::-1][:3]
        idcg = np.sum((2**igains - 1) / np.log2(np.arange(2, 2 + len(igains))))
        
        if idcg > 0:
            ndcgs.append(dcg / idcg)
            
    return np.mean(ndcgs) if ndcgs else 0.0


def tune_with_optuna(train, valid, features, cat_cols, n_trials: int = 50):
    """Use Optuna to find optimal hyperparameters."""
    import optuna
    from optuna.samplers import TPESampler
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 10],
            "verbosity": -1,
            "feature_pre_filter": False,
            
            # Tunable parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        
        try:
            model = train_with_params(train, valid, features, cat_cols, params)
            ndcg = compute_ndcg(model, valid, features)
            return ndcg
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="f1_lgb_tuning"
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value


def main(tune: bool = False, n_trials: int = 50):
    """Train the ranking model."""
    df = load_feature_table()
    
    print(f"📊 Loaded {len(df):,} rows from {FE_PATH.name}")
    print(f"   Years: {df['event_year'].min()} - {df['event_year'].max()}")
    print(f"   Races: {df['race_id'].nunique()}")

    train, valid, features, cat_cols = prepare_data(df)
    
    print(f"\n🔀 Split: Train {len(train):,} rows | Valid {len(valid):,} rows")
    print(f"   Features: {len(features)} ({len(cat_cols)} categorical)")

    if tune:
        print(f"\n🔍 Running Optuna hyperparameter tuning ({n_trials} trials)...")
        print("   This may take 10-30 minutes...\n")
        
        best_params, best_score = tune_with_optuna(train, valid, features, cat_cols, n_trials)
        
        print(f"\n✨ Best NDCG@3: {best_score:.4f}")
        print("📋 Best parameters:")
        for k, v in best_params.items():
            print(f"   {k}: {v}")
        
        # Train final model with best params
        final_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 10],
            "verbosity": -1,
            "feature_pre_filter": False,
            **best_params
        }
    else:
        print("\n🏋️ Training with default parameters...")
        final_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 10],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "verbosity": -1,
        }

    # Final training with logging
    print("\n🏋️ Training final model...")
    
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

    model = lgb.train(
        final_params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )

    # Save model and metadata
    model_path = MODEL_DIR / "ranker_lgb.txt"
    model.save_model(str(model_path))
    
    meta = {
        "features": features,
        "cat_cols": cat_cols,
        "params": final_params,
        "tuned": tune,
    }
    meta_path = MODEL_DIR / "ranker_meta.joblib"
    joblib.dump(meta, meta_path)
    
    print(f"\n✅ Saved model → {model_path}")
    print(f"✅ Saved meta  → {meta_path}")

    # Final validation score
    ndcg = compute_ndcg(model, valid, features)
    print(f"\n📈 Final Validation NDCG@3: {ndcg:.4f}")
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])[:10]
    print("\n🏆 Top 10 Features:")
    for feat, imp in feat_imp:
        print(f"   {feat}: {imp:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 prediction model")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    args = parser.parse_args()
    
    main(tune=args.tune, n_trials=args.n_trials)
