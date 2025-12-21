from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def american_to_implied_prob(odds: float) -> float:
    """
    Converts American odds to implied probability (no vig removal).
    +2000 -> 1 / (20 + 1) = 0.0476
    -150 -> 150 / (150 + 100) = 0.60
    """
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def implied_prob_to_american(p: float) -> float:
    """
    Converts probability to American odds.
    0.0476 -> +2000
    0.60 -> -150
    """
    p = float(p)
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p < 0.5:
        return (100.0 / p) - 100.0
    return -((100.0 * p) / (1.0 - p))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    ex = np.exp(x)
    s = np.nansum(ex)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(ex, 1.0 / len(ex))
    return ex / s


def attach_market_odds(df: pd.DataFrame, odds_csv: Path) -> pd.DataFrame:
    """
    Expects odds_csv columns:
    - PLAYER (preferred) or PLAYER_ID
    - american_odds
    """
    market = pd.read_csv(odds_csv)

    if "american_odds" not in market.columns:
        raise ValueError("odds file must contain column: american_odds")

    out = df.copy()

    if "PLAYER_ID" in market.columns and "PLAYER_ID" in out.columns:
        out = out.merge(market[["PLAYER_ID", "american_odds"]], on="PLAYER_ID", how="left")
    elif "PLAYER" in market.columns and "PLAYER" in out.columns:
        out = out.merge(market[["PLAYER", "american_odds"]], on="PLAYER", how="left")
    else:
        raise ValueError("odds file must contain PLAYER_ID or PLAYER to join")

    out["implied_win_prob"] = out["american_odds"].apply(
        lambda o: american_to_implied_prob(o) if pd.notna(o) else np.nan
    )

    return out


def compute_model_win_probs(
    df: pd.DataFrame,
    score_col: str = "model_strength",
    strength_from: str = "fit_final",
    k_fit: float = 1.0,
    temp: float = 0.25,
) -> pd.DataFrame:
    """
    Builds a simple model strength score and converts to win probabilities via softmax.

    strength = overall_talent + k_fit * fit_final

    temp smaller -> more concentrated favorites
    temp larger -> more spread out
    """
    out = df.copy()

    if "overall_talent" not in out.columns:
        raise ValueError("DataFrame missing overall_talent. Run estimate_all_talents first.")
    if strength_from not in out.columns:
        raise ValueError(f"DataFrame missing {strength_from}. Did you build fit_final?")

    out[score_col] = pd.to_numeric(out["overall_talent"], errors="coerce").fillna(0.0) + \
                     float(k_fit) * pd.to_numeric(out[strength_from], errors="coerce").fillna(0.0)

    # Softmax over field
    scores = out[score_col].to_numpy(dtype=float)
    probs = softmax(scores / float(temp))
    out["model_win_prob"] = probs

    # Fair odds from model
    out["model_fair_american"] = out["model_win_prob"].apply(implied_prob_to_american)

    return out


def find_betting_edges(
    df: pd.DataFrame,
    min_edge_pp: float = 1.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Edge is model_win_prob - implied_win_prob.
    min_edge_pp is in percentage points (1.0 = 1%).
    """
    out = df.copy()
    if "model_win_prob" not in out.columns or "implied_win_prob" not in out.columns:
        raise ValueError("Need model_win_prob and implied_win_prob columns.")

    out["edge"] = out["model_win_prob"] - out["implied_win_prob"]
    out["edge_pp"] = out["edge"] * 100.0

    edges = out[pd.notna(out["implied_win_prob"])].copy()
    edges = edges[edges["edge_pp"] >= float(min_edge_pp)]
    edges = edges.sort_values("edge_pp", ascending=False).head(top_n)

    cols = [
        "PLAYER", "PLAYER_ID",
        "world_rank",
        "overall_talent",
        "fit_final",
        "american_odds",
        "implied_win_prob",
        "model_win_prob",
        "model_fair_american",
        "edge_pp",
    ]
    cols = [c for c in cols if c in edges.columns]
    return edges[cols].reset_index(drop=True)
