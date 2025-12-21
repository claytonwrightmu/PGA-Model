from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_HISTORICAL, DATA_RAW, ensure_dirs
from src.data_engine.load_raw_stats import load_all_player_stats
from src.data_engine.validate_data import validate_player_data
from src.data_engine.calculate_talent import estimate_all_talents
from src.course_engine.calculate_fits import calculate_all_course_fits
from src.course_engine.validate_fits import load_historical_results, validate_fits

logger = logging.getLogger(__name__)


def main(archetype_for_validation: str = "classic_ballstriking") -> None:
    """
    Backtests whether talent and a chosen archetype fit correlate with results.
    Real version later maps each event_id to an archetype.
    """
    ensure_dirs()

    df = load_all_player_stats(year=2025, data_path=DATA_RAW, include_history=True)
    report = validate_player_data(df)
    if not report["passed"]:
        raise RuntimeError(f"Validation failed: {report['errors']}")

    df = estimate_all_talents(df)
    df = calculate_all_course_fits(df)

    results_all = []
    for year in (2023, 2024):
        res_path = DATA_HISTORICAL / f"{year}_results.csv"
        res = load_historical_results(res_path)
        metrics = validate_fits(df, res, archetype_key=archetype_for_validation)
        metrics["year"] = year
        results_all.append(metrics)

    print("\nBACKTEST SUMMARY")
    for m in results_all:
        year = m["year"]
        print(f"\nYear {year}")
        print("overall_talent:", m["overall_talent"])
        fit_col = f"fit_{archetype_for_validation}"
        print(fit_col + ":", m[fit_col])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
