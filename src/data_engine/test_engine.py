from __future__ import annotations

import logging
from pathlib import Path

from config import DATA_RAW, ensure_dirs
from src.data_engine.load_raw_stats import load_all_player_stats
from src.data_engine.validate_data import validate_player_data
from src.data_engine.calculate_talent import estimate_all_talents
from src.course_engine.calculate_fits import calculate_all_course_fits
from src.analysis.sleeper_detector import find_sleepers
from src.analysis.fragility_analyzer import find_fragile_players

logger = logging.getLogger(__name__)


def run_engine_smoke_test(year: int = 2025, data_path: Path = DATA_RAW) -> None:
    ensure_dirs()
    df = load_all_player_stats(year=year, data_path=data_path, include_history=True)

    report = validate_player_data(df)
    if not report["passed"]:
        raise RuntimeError(f"Validation failed: {report['errors']}")

    df = estimate_all_talents(df)
    df = calculate_all_course_fits(df)

    sleepers = find_sleepers(df, "bomber_paradise", top_n=5)
    fades = find_fragile_players(df, "accuracy_premium", top_n=5)

    logger.info("Smoke test passed")
    logger.info("Top sleepers:\n%s", sleepers[["PLAYER", "value_score"]].to_string(index=False))
    logger.info("Top fades:\n%s", fades[["PLAYER", "fragility_score"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_engine_smoke_test()
