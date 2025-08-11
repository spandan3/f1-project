import fastf1
import pandas as pd
from backend.data.utils import save_df

fastf1.Cache.enable_cache('f1_cache')

def main():
    session = fastf1.get_session(2023, 'Monza', 'Q')
    session.load()

    laps = session.laps
    save_df(laps, 'data/raw/laps_monza_2023_q.parquet')

if __name__ == "__main__":
    main()