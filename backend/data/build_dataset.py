import pandas as pd, numpy as np
from pathlib import Path

# Base project directory (two levels up from backend/data/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_LAPS = BASE_DIR / "data" / "raw" / "laps.parquet"
RAW_RES = BASE_DIR / "data" / "raw" / "results.parquet"
FE_DIR = BASE_DIR / "data" / "fe"
FE_DIR.mkdir(parents=True, exist_ok=True)

def make_pre_race_table() -> pd.DataFrame:
    laps = pd.read_parquet(RAW_LAPS)
    res  = pd.read_parquet(RAW_RES)

    # Use race results (R) as labels; use qualifying (Q) for grid/quali gap
    quali = res[res["session_type"]=="Q"][["DriverNumber","Driver","TeamName","event_name","event_year","Position","Q1","Q2","Q3"]]
    quali = quali.rename(columns={"Position":"grid_pos"})
    race  = res[res["session_type"]=="R"][["DriverNumber","Driver","TeamName","event_name","event_year","Position","Status"]]
    race = race.rename(columns={"Position":"finish_pos"})

    # Quali best time per driver (as seconds)
    def to_s(x):
        if pd.isna(x): return np.nan
        m, s = str(x).split(":")
        return int(m)*60+float(s)

    quali["best_q"] = quali[["Q3","Q2","Q1"]].apply(
        lambda r: pd.Series([to_s(r["Q3"]), to_s(r["Q2"]), to_s(r["Q1"])]).min(), axis=1
    )

    # Gap to pole
    pole = quali.groupby(["event_year","event_name"])["best_q"].transform("min")
    quali["quali_gap_s"] = quali["best_q"] - pole

    # Join race labels
    df = pd.merge(
        race,
        quali[["DriverNumber","event_year","event_name","grid_pos","quali_gap_s"]],
        on=["DriverNumber","event_year","event_name"], how="left"
    )

    # Driver & constructor rolling form (last 3 races prior to this event)
    df = df.sort_values(["Driver","event_year","event_name"])
    grp = df.groupby("Driver", group_keys=False)
    df["driver_last3_avg_finish"] = grp["finish_pos"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    df["driver_last3_dnfs"] = grp.apply(
        lambda g: (g["finish_pos"].isna() | (g["finish_pos"]<=0)).shift().rolling(3, min_periods=1).sum()
    ).reset_index(level=0, drop=True)

    g2 = df.groupby("TeamName", group_keys=False)
    df["team_last3_avg_finish"] = g2["finish_pos"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())

    # Simple track descriptors (placeholders)
    df["track_overtake_idx"] = 0.5
    df["pit_loss_s"] = 22.0

    # Weather (placeholder)
    df["is_wet_flag"] = 0

    # Label + relevance (for LambdaMART)
    df["relevance"] = (df["finish_pos"]==1)*3 + (df["finish_pos"]==2)*2 + (df["finish_pos"]==3)*1

    # Group id per race
    df["race_id"] = df["event_year"].astype(str) + "_" + df["event_name"].astype(str)

    # Minimal features
    features = ["grid_pos","quali_gap_s","driver_last3_avg_finish","team_last3_avg_finish",
                "track_overtake_idx","pit_loss_s","is_wet_flag"]
    keep = ["race_id","Driver","DriverNumber","TeamName","finish_pos","relevance"] + features
    out = df[keep].dropna(subset=["grid_pos","quali_gap_s"], how="any")

    # Save to data/fe/
    out_path = FE_DIR / "standings_train.parquet"
    out.to_parquet(out_path, index=False)
    print(f"✅ Saved features to {out_path}")
    return out

if __name__=="__main__":
    make_pre_race_table()
