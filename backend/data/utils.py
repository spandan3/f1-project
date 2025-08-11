import pandas as pd
import os

def save_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved dataframe to {path}")

def load_df(path: str) -> pd.DataFrame:
    print(f"📂 Loading dataframe from {path}")
    return pd.read_parquet(path)