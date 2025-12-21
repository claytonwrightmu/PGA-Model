from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from config import DATA_RAW, OUTPUT_PREDICTIONS
from src.utils.helpers import stamp

from src.data_engine.load_raw_stats import load_all_player_stats
from src.data_engine.validate_data import validate_player_data
from src.data_engine.calculate_talent import estimate_all_talents
from src.course_engine.calculate_fits import calculate_all_course_fits, calculate_tournament_fit

from src.analysis.sleeper_detector import find_sleepers
from src.analysis.fragility_analyzer import find_fragile_players
from src.analysis.condition_matcher import apply_condition_modifiers

from src.analysis.betting_edge import (
    attach_market_odds,
    compute_model_win_probs,
    find_betting_edges,
)

from predictions.tournament_context import get_tournament_context

logger = logging.getLogger(__name__)


def _ensure_dirs():
    OUTPUT_PREDICTIONS.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionConfig:
    tournament: str = "template_mix"
    year: int = 2025
    data_path: Path = DATA_RAW

    top_n_sleepers: int = 10
    top_n_value_plays: int = 15
    top_n_fades: int = 10
    top_n_edges: int = 25

    # Sleeper thresholds (strict)
    sleeper_min_course_fit: float = 0.20
    sleeper_max_public_rank: int = 50
    sleeper_min_talent_percentile: float = 60.0

    # Value plays thresholds (broader)
    value_min_course_fit: float = 0.10
    value_max_public_rank: int = 25
    value_min_talent_percentile: float = 55.0

    # Fades archetype (can be specific)
    archetype_fades: str = "accuracy_premium"

    # Betting edge config
    markets_dir: Path = Path("data/processed/markets")
    odds_join_key: str = "PLAYER"  # keep as PLAYER unless you switch to PLAYER_ID
    odds_filename_override: Optional[str] = None  # e.g. "genesis_2026_odds.csv"

    # Probability calibration knobs (placeholder until backtesting)
    k_fit: float = 1.0
    temp: float = 0.25
    min_edge_pp: float = 1.0  # minimum edge in percentage points


def _run_sleepers_on_fit(
    df: pd.DataFrame,
    fit_col: str,
    archetype_key_for_output: str,
    top_n: int,
    min_fit: float,
    max_public_rank: int,
    min_talent_percentile: float,
) -> pd.DataFrame:
    """
    Reuse sleeper engine by temporarily mapping fit_col to expected name.
    Keeps scoring logic consistent.
    """
    temp = df.copy()
    temp[f"fit_{archetype_key_for_output}"] = pd.to_numeric(temp[fit_col], errors="coerce")

    out = find_sleepers(
        temp,
        course_archetype=archetype_key_for_output,
        top_n=top_n,
        min_course_fit=min_fit,
        max_public_rank=max_public_rank,
    )

    # Ensure percentile filter
    if "talent_percentile" in out.columns:
        out = out[out["talent_percentile"] >= float(min_talent_percentile)].reset_index(drop=True)

    return out


def _odds_path(cfg: PredictionConfig, ctx_name: str) -> Path:
    if cfg.odds_filename_override:
        return cfg.markets_dir / cfg.odds_filename_override
    return cfg.markets_dir / f"{ctx_name}_odds.csv"


def run_prediction(cfg: PredictionConfig = PredictionConfig()) -> Dict[str, Any]:
    _ensure_dirs()

    ctx = get_tournament_context(cfg.tournament, year=cfg.year)
    logger.info(
        "Tournament context: %s | mix=%s | conditions=%s",
        ctx.name,
        ctx.archetype_mix,
        ctx.conditions,
    )

    # 1) Load + validate
    df = load_all_player_stats(year=cfg.year, data_path=cfg.data_path, include_history=True)
    report = validate_player_data(df)
    if not report["passed"]:
        raise RuntimeError(f"Validation failed: {report['errors']}")

    # 2) Talent + archetype fits
    df = estimate_all_talents(df)
    df = calculate_all_course_fits(df)

    # 3) Tournament blended fit
    df = calculate_tournament_fit(df, ctx.archetype_mix, out_col="fit_tournament")

    # 4) Condition deltas and final fit
    df = apply_condition_modifiers(df, ctx.conditions)
    df["fit_final"] = df["fit_tournament"].fillna(0.0) + df["cond_fit_delta"].fillna(0.0)

    # 5) Sleepers/value plays using fit_final
    sleepers = _run_sleepers_on_fit(
        df=df,
        fit_col="fit_final",
        archetype_key_for_output="tournament",
        top_n=cfg.top_n_sleepers,
        min_fit=cfg.sleeper_min_course_fit,
        max_public_rank=cfg.sleeper_max_public_rank,
        min_talent_percentile=cfg.sleeper_min_talent_percentile,
    )

    value_plays = _run_sleepers_on_fit(
        df=df,
        fit_col="fit_final",
        archetype_key_for_output="tournament",
        top_n=cfg.top_n_value_plays,
        min_fit=cfg.value_min_course_fit,
        max_public_rank=cfg.value_max_public_rank,
        min_talent_percentile=cfg.value_min_talent_percentile,
    )

    # 6) Fades (specific archetype)
    fades = find_fragile_players(df, cfg.archetype_fades, top_n=cfg.top_n_fades)

    # 7) Betting edges (requires odds file)
    odds_file = _odds_path(cfg, ctx.name)
    edges = pd.DataFrame()
    matched_odds = 0

    if odds_file.exists():
        df = attach_market_odds(df, odds_file)
        matched_odds = int(df["american_odds"].notna().sum()) if "american_odds" in df.columns else 0

        if matched_odds == 0:
            logger.warning(
                "Odds file found (%s) but 0 rows matched. Check spelling in PLAYER column.",
                odds_file,
            )
        else:
            df = compute_model_win_probs(df, k_fit=cfg.k_fit, temp=cfg.temp)
            edges = find_betting_edges(
                df,
                min_edge_pp=cfg.min_edge_pp,
                top_n=cfg.top_n_edges,
            )
            logger.info("✓ Betting edge computed (matched odds for %s players)", matched_odds)
    else:
        logger.warning("No odds file found at %s, skipping betting edge output", odds_file)

    # 8) Save outputs
    run_id = stamp("pred")
    master_path = OUTPUT_PREDICTIONS / f"{run_id}_{ctx.name}_master.csv"
    sleepers_path = OUTPUT_PREDICTIONS / f"{run_id}_{ctx.name}_sleepers.csv"
    value_path = OUTPUT_PREDICTIONS / f"{run_id}_{ctx.name}_value_plays.csv"
    fades_path = OUTPUT_PREDICTIONS / f"{run_id}_{ctx.name}_fades_{cfg.archetype_fades}.csv"
    edges_path = OUTPUT_PREDICTIONS / f"{run_id}_{ctx.name}_betting_edges.csv"

    df.to_csv(master_path, index=False)
    sleepers.to_csv(sleepers_path, index=False)
    value_plays.to_csv(value_path, index=False)
    fades.to_csv(fades_path, index=False)

    logger.info("✓ Saved master to %s", master_path)
    logger.info("✓ Saved sleepers to %s", sleepers_path)
    logger.info("✓ Saved value plays to %s", value_path)
    logger.info("✓ Saved fades to %s", fades_path)

    if len(edges) > 0:
        edges.to_csv(edges_path, index=False)
        logger.info("✓ Saved betting edges to %s", edges_path)
    else:
        logger.info("No betting edges saved (need matched odds + edge threshold).")

    return {
        "df": df,
        "sleepers": sleepers,
        "value_plays": value_plays,
        "fades": fades,
        "betting_edges": edges,
        "tournament_context": ctx,
        "validation_report": report,
        "odds_file": str(odds_file),
        "matched_odds_rows": matched_odds,
        "paths": {
            "master": master_path,
            "sleepers": sleepers_path,
            "value_plays": value_path,
            "fades": fades_path,
            "betting_edges": edges_path if len(edges) > 0 else None,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_prediction()
