from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_HISTORICAL, ensure_dirs


def main() -> None:
    """
    This script is a placeholder for scraping.
    For now, it just validates that 2023_results.csv and 2024_results.csv exist
    and have required columns.
    """
    ensure_dirs()
    for year in (2023, 2024):
        p = DATA_HISTORICAL / f"{year}_results.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

        df = pd.read_csv(p)
        required = ["event_id", "PLAYER_ID", "finish_position"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} missing columns: {missing}")

    print("✓ Historical results files exist and have required columns")


if __name__ == "__main__":
    main()
