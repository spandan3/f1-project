"""
Database loader: Converts Parquet files to SQLite for querying.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd
from typing import Optional

from ..ml.common import BASE_DIR, RAW_DIR


DB_PATH = BASE_DIR / "data" / "f1_chatbot.db"


def load_parquet_to_sqlite(force_rebuild: bool = False) -> Path:
    """
    Convert Parquet files to SQLite database.
    
    Creates tables:
    - results: Race and qualifying results
    - laps: Lap-by-lap data
    - weather: Weather data per session
    
    Args:
        force_rebuild: If True, drop existing database and rebuild
        
    Returns:
        Path to SQLite database
    """
    if DB_PATH.exists() and not force_rebuild:
        print(f"✅ Database already exists at {DB_PATH}")
        return DB_PATH
    
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"🗑️  Removed existing database")
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # === RESULTS TABLE ===
    results_path = RAW_DIR / "results.parquet"
    if results_path.exists():
        print("📊 Loading results...")
        df_res = pd.read_parquet(results_path)
        
        # Normalize column names
        column_map = {
            "Driver": "Driver",
            "Abbreviation": "Driver",
            "DriverId": "Driver",
            "TeamName": "TeamName",
            "Team": "TeamName",
            "Position": "Position",
            "GridPosition": "GridPosition",
            "DriverNumber": "DriverNumber",
        }
        
        # Map columns
        for old_col, new_col in column_map.items():
            if old_col in df_res.columns and new_col not in df_res.columns:
                df_res[new_col] = df_res[old_col]
        
        # Ensure required columns exist
        required_cols = ["event_year", "event_name", "session_type"]
        for col in required_cols:
            if col not in df_res.columns:
                # Try alternative names
                alt = {"event_year": "EventYear", "event_name": "EventName", 
                       "session_type": "SessionType"}
                if alt[col] in df_res.columns:
                    df_res[col] = df_res[alt[col]]
        
        # Select and clean columns
        keep_cols = [
            "event_year", "event_name", "session_type",
            "Driver", "TeamName", "DriverNumber",
            "Position", "GridPosition",
        ]
        # Add optional columns if they exist
        optional_cols = ["Q1", "Q2", "Q3", "FastestLap", "FastestLapTime", 
                        "FastestLapSpeed", "Status", "Points", "Time"]
        for col in optional_cols:
            if col in df_res.columns:
                keep_cols.append(col)
        
        df_res = df_res[[c for c in keep_cols if c in df_res.columns]].copy()
        df_res.to_sql("results", conn, if_exists="replace", index=False)
        print(f"   ✓ {len(df_res):,} rows")
        
        # Create indexes
        conn.execute("CREATE INDEX idx_results_year_name ON results(event_year, event_name)")
        conn.execute("CREATE INDEX idx_results_driver ON results(Driver)")
        conn.execute("CREATE INDEX idx_results_session ON results(session_type)")
    else:
        print("⚠️  results.parquet not found")
    
    # === LAPS TABLE ===
    laps_path = RAW_DIR / "laps.parquet"
    if laps_path.exists():
        print("📊 Loading laps...")
        df_laps = pd.read_parquet(laps_path)
        
        # Normalize columns
        for old_col, new_col in column_map.items():
            if old_col in df_laps.columns and new_col not in df_laps.columns:
                df_laps[new_col] = df_laps[old_col]
        
        # Ensure required columns
        for col in required_cols:
            if col not in df_laps.columns:
                alt = {"event_year": "EventYear", "event_name": "EventName", 
                       "session_type": "SessionType"}
                if alt[col] in df_laps.columns:
                    df_laps[col] = df_laps[alt[col]]
        
        # Select key columns
        keep_cols = [
            "event_year", "event_name", "session_type",
            "Driver", "TeamName", "DriverNumber",
            "LapNumber", "LapTime", "Position",
        ]
        optional_lap_cols = ["Sector1Time", "Sector2Time", "Sector3Time", 
                            "Compound", "TyreLife", "IsPersonalBest", "IsFastest"]
        for col in optional_lap_cols:
            if col in df_laps.columns:
                keep_cols.append(col)
        
        df_laps = df_laps[[c for c in keep_cols if c in df_laps.columns]].copy()
        
        # Convert LapTime to string for SQLite (store as TEXT)
        if "LapTime" in df_laps.columns:
            df_laps["LapTime"] = df_laps["LapTime"].astype(str)
        
        df_laps.to_sql("laps", conn, if_exists="replace", index=False)
        print(f"   ✓ {len(df_laps):,} rows")
        
        conn.execute("CREATE INDEX idx_laps_year_name ON laps(event_year, event_name)")
        conn.execute("CREATE INDEX idx_laps_driver ON laps(Driver)")
        conn.execute("CREATE INDEX idx_laps_session ON laps(session_type)")
    else:
        print("⚠️  laps.parquet not found")
    
    # === WEATHER TABLE ===
    weather_path = RAW_DIR / "weather.parquet"
    if weather_path.exists():
        print("📊 Loading weather...")
        df_wx = pd.read_parquet(weather_path)
        
        # Ensure required columns
        for col in required_cols:
            if col not in df_wx.columns:
                alt = {"event_year": "EventYear", "event_name": "EventName", 
                       "session_type": "SessionType"}
                if alt[col] in df_wx.columns:
                    df_wx[col] = df_wx[alt[col]]
        
        df_wx.to_sql("weather", conn, if_exists="replace", index=False)
        print(f"   ✓ {len(df_wx):,} rows")
        
        conn.execute("CREATE INDEX idx_weather_year_name ON weather(event_year, event_name)")
    else:
        print("⚠️  weather.parquet not found")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Database created at {DB_PATH}")
    return DB_PATH


def get_db_schema() -> str:
    """
    Generate schema description for LLM prompt.
    
    Returns:
        Formatted schema string
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run load_parquet_to_sqlite() first.")
    
    conn = sqlite3.connect(str(DB_PATH))
    
    schema_parts = ["# F1 Database Schema\n"]
    
    # Get table names
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    
    for (table_name,) in tables:
        schema_parts.append(f"\n## Table: {table_name}\n")
        schema_parts.append("Columns:\n")
        
        # Get column info
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        for _, col_name, col_type, not_null, default, pk in cols:
            schema_parts.append(f"  - {col_name} ({col_type})")
            if pk:
                schema_parts[-1] += " [PRIMARY KEY]"
            if not_null:
                schema_parts[-1] += " [NOT NULL]"
            schema_parts[-1] += "\n"
        
        # Get sample data
        sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
        if sample:
            schema_parts.append("\nSample rows:\n")
            col_names = [desc[0] for desc in conn.execute(f"SELECT * FROM {table_name} LIMIT 1").description]
            for row in sample:
                row_str = " | ".join(str(v) if v is not None else "NULL" for v in row)
                schema_parts.append(f"  {row_str}\n")
    
    conn.close()
    
    return "".join(schema_parts)


if __name__ == "__main__":
    load_parquet_to_sqlite(force_rebuild=True)
    print("\n" + "="*60)
    print(get_db_schema())

