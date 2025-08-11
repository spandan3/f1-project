# app.py
import argparse, os, joblib, numpy as np, pandas as pd, lightgbm as lgb
from backend.data.build_dataset import build_dataset, build_single_race_features
from backend.data.utils import ensure_dirs

PROCESSED_DIR = "data/processed"
MODEL_DIR = os.path.join(PROCESSED_DIR, "models")

FEATURES = [
    "quali_pos","grid_pos","driver_form_3","driver_form_5",
    "team_form_3","team_form_5",
    "air_temp","track_temp","humidity","wind_speed","wind_dir","pressure","rainfall",
    "track_id","team_id","driver_id","round"
]

def train_models(df, test_season):
    train_df = df[df["season"] != test_season].copy()
    test_df  = df[df["season"] == test_season].copy()

    X_tr, y_tr = train_df[FEATURES], train_df["scored_points"].astype(int)
    X_te, y_te = test_df[FEATURES], test_df["scored_points"].astype(int)

    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=600, learning_rate=0.03,
        num_leaves=63, subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:,1]
    binary_auc = float(lgb.metric.auc(y_te, proba)) if len(np.unique(y_te))>1 else None
    binary_acc = float((proba>=0.5).astype(int).mean())

    # Ranking
    tr_groups = train_df.groupby(["season","round"]).size().values.tolist()
    te_groups = test_df.groupby(["season","round"]).size().values.tolist()

    rank_train = lgb.Dataset(train_df[FEATURES], label=train_df["relevance"], group=tr_groups, free_raw_data=False)
    rank_valid = lgb.Dataset(test_df[FEATURES],   label=test_df["relevance"], group=te_groups, free_raw_data=False)

    rank_params = {
        "objective":"lambdarank","metric":"ndcg","ndcg_eval_at":[5,10,20],
        "learning_rate":0.05,"num_leaves":63,"min_data_in_leaf":20,"verbosity":-1,"random_state":42
    }
    ranker = lgb.train(rank_params, rank_train, valid_sets=[rank_valid], num_boost_round=800, early_stopping_rounds=50, verbose_eval=False)

    metrics = {"binary_auc": binary_auc, "binary_acc": binary_acc}
    return clf, ranker, metrics

def cmd_build(args):
    seasons = list(range(args.start_season, args.end_season + 1))
    df = build_dataset(seasons, cache_dir=args.cache)
    ensure_dirs(PROCESSED_DIR)
    outp = os.path.join(PROCESSED_DIR, "dataset.parquet")
    df.to_parquet(outp, index=False)
    print(f"Saved dataset → {outp}   shape={df.shape}")

def cmd_train(args):
    ds_path = os.path.join(PROCESSED_DIR, "dataset.parquet")
    df = pd.read_parquet(ds_path)
    ensure_dirs(MODEL_DIR)
    clf, ranker, metrics = train_models(df, test_season=args.test_season)
    joblib.dump(clf, os.path.join(MODEL_DIR, "points_classifier.joblib"))
    ranker.save_model(os.path.join(MODEL_DIR, "ranker.txt"))
    print("Saved models.")
    print("Metrics:", metrics)

def _load_models():
    clf = joblib.load(os.path.join(MODEL_DIR, "points_classifier.joblib"))
    ranker = lgb.Booster(model_file=os.path.join(MODEL_DIR, "ranker.txt"))
    return clf, ranker

def cmd_predict(args):
    clf, ranker = _load_models()
    features = build_single_race_features(args.season, args.round, cache_dir=args.cache)
    # ensure feature cols exist for predict
    features = features.copy()
    for c in FEATURES:
        if c not in features.columns:
            features[c] = 0
    features["points_proba"] = clf.predict_proba(features[FEATURES])[:,1]
    features["rank_score"] = ranker.predict(features[FEATURES])
    features = features.sort_values("rank_score", ascending=False)
    features["predicted_finish_pos"] = range(1, len(features)+1)

    ensure_dirs(PROCESSED_DIR)
    outp = os.path.join(PROCESSED_DIR, f"predictions_{args.season}_{args.round}.csv")
    cols = ["season","round","driver","team","quali_pos","grid_pos","points_proba","rank_score","predicted_finish_pos"]
    features[cols].to_csv(outp, index=False)
    print(f"Saved predictions → {outp}")

def main():
    p = argparse.ArgumentParser(description="F1 LightGBM pipeline")
    sub = p.add_subparsers(required=True)

    b = sub.add_parser("build", help="Build dataset")
    b.add_argument("--start-season", type=int, default=2018)
    b.add_argument("--end-season", type=int, default=2024)
    b.add_argument("--cache", type=str, default="./f1_cache")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="Train models")
    t.add_argument("--test-season", type=int, default=2024)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Predict a specific race (after quali)")
    pr.add_argument("--season", type=int, required=True)
    pr.add_argument("--round", type=int, required=True)
    pr.add_argument("--cache", type=str, default="./f1_cache")
    pr.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
