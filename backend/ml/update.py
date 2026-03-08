# backend/ml/update.py
"""
Rolling update script for 2026 season.
After each race: fetch new data → rebuild features → retrain model.
"""
from __future__ import annotations

import argparse
import sys

from .fetch_data import fetch_single_race, get_available_races
from .build_features import build_features
from .trainer import main as train_model


def update_after_race(year: int, event_name: str, retrain: bool = True) -> dict:
    """
    Full update pipeline after a race completes.
    
    1. Fetch new race data from FastF1
    2. Rebuild feature dataset
    3. Retrain model on all data
    
    Args:
        year: Race year (e.g., 2026)
        event_name: GP name (e.g., "Bahrain Grand Prix")
        retrain: Whether to retrain the model
        
    Returns:
        Status dict
    """
    print("=" * 60)
    print(f"🏎️ UPDATING: {year} {event_name}")
    print("=" * 60)
    
    # Step 1: Fetch race data (need both Q and R sessions after race completes)
    print("\n📥 Step 1: Fetching race data...")
    fetch_result = fetch_single_race(year, event_name, require_race=True)
    
    if not fetch_result.get("ok"):
        print(f"❌ Failed to fetch race: {fetch_result.get('error')}")
        return {"ok": False, "step": "fetch", "error": fetch_result.get("error")}
    
    # Step 2: Rebuild features
    print("\n🔧 Step 2: Rebuilding features...")
    try:
        features_path = build_features(output_name="features.parquet")
    except Exception as e:
        print(f"❌ Failed to build features: {e}")
        return {"ok": False, "step": "features", "error": str(e)}
    
    # Step 3: Retrain model
    if retrain:
        print("\n🧠 Step 3: Retraining model...")
        try:
            train_model()
        except Exception as e:
            print(f"❌ Failed to train model: {e}")
            return {"ok": False, "step": "train", "error": str(e)}
    else:
        print("\n⏭️ Step 3: Skipping retrain (--no-retrain)")
    
    print("\n" + "=" * 60)
    print("✅ UPDATE COMPLETE")
    print("=" * 60)
    
    return {
        "ok": True,
        "year": year,
        "event": event_name,
        "features_path": str(features_path),
        "retrained": retrain
    }


def list_upcoming_races(year: int):
    """Show available races for a year."""
    races = get_available_races(year)
    if not races:
        print(f"No races found for {year}")
        return
    
    print(f"\n🏁 {year} F1 Calendar ({len(races)} races):")
    for i, race in enumerate(races, 1):
        print(f"   {i:2d}. {race}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update F1 prediction system after a race",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update after Bahrain GP
  python -m backend.ml.update --year 2026 --race "Bahrain Grand Prix"
  
  # Update without retraining (just add data)
  python -m backend.ml.update --year 2026 --race "Saudi Arabian Grand Prix" --no-retrain
  
  # List available races
  python -m backend.ml.update --list 2026
        """
    )
    parser.add_argument("--year", type=int, help="Race year (e.g., 2026)")
    parser.add_argument("--race", type=str, help="GP name (e.g., 'Bahrain Grand Prix')")
    parser.add_argument("--no-retrain", action="store_true", help="Skip model retraining")
    parser.add_argument("--list", type=int, metavar="YEAR", help="List races for a year")
    
    args = parser.parse_args()
    
    if args.list:
        list_upcoming_races(args.list)
    elif args.year and args.race:
        result = update_after_race(args.year, args.race, retrain=not args.no_retrain)
        sys.exit(0 if result["ok"] else 1)
    else:
        parser.print_help()
        print("\n❌ Error: Provide --year and --race, or use --list YEAR")
        sys.exit(1)

