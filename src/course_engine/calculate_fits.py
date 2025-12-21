from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.course_engine.define_archetypes import get_archetypes

logger = logging.getLogger(__name__)


def _skill_series(df: pd.DataFrame, skill: str) -> pd.Series:
    """
    Returns best available series for a skill:
    prefers {skill}_talent, falls back to raw {skill}.
    """
    talent_col = f"{skill}_talent"
    if talent_col in df.columns:
        return pd.to_numeric(df[talent_col], errors="coerce")
    if skill in df.columns:
        return pd.to_numeric(df[skill], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def calculate_fit_for_archetype(df: pd.DataFrame, archetype_def: Dict) -> pd.Series:
    """
    Fit = sum(skill_value * weight).
    Missing skills contribute 0.
    """
    weights = archetype_def.get("weights", {})
    fit = pd.Series(0.0, index=df.index, dtype=float)

    for skill, w in weights.items():
        s = _skill_series(df, skill).fillna(0.0)
        fit = fit + s * float(w)

    return fit


def calculate_all_course_fits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fit_{archetype} columns for every archetype.
    Also adds best_fit_score and best_archetype.
    """
    out = df.copy()
    archetypes = get_archetypes()

    for key, meta in archetypes.items():
        out[f"fit_{key}"] = calculate_fit_for_archetype(out, meta)

    fit_cols = [c for c in out.columns if c.startswith("fit_")]
    if fit_cols:
        out["best_fit_score"] = out[fit_cols].max(axis=1, skipna=True)
        out["best_archetype"] = (
            out[fit_cols].idxmax(axis=1).str.replace("fit_", "", regex=False)
        )

    logger.info("✓ Course fits calculated for %s archetypes", len(archetypes))
    return out


def calculate_tournament_fit(
    df: pd.DataFrame,
    archetype_mix: Dict[str, float],
    out_col: str = "fit_tournament",
) -> pd.DataFrame:
    """
    Blends archetype fits into a tournament-specific fit column.

    archetype_mix example:
        {"bomber_paradise": 0.6, "target_golf": 0.4}

    Requires calculate_all_course_fits() to have run first.
    """
    out = df.copy()

    mix = {k: float(v) for k, v in archetype_mix.items() if float(v) > 0}
    if not mix:
        out[out_col] = np.nan
        return out

    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items()}

    missing = [k for k in mix.keys() if f"fit_{k}" not in out.columns]
    if missing:
        raise ValueError(
            f"Missing fit columns for archetypes: {missing}. "
            "Did you run calculate_all_course_fits()?"
        )

    out[out_col] = 0.0
    for k, w in mix.items():
        out[out_col] = out[out_col] + out[f"fit_{k}"].fillna(0.0) * w

    return out
