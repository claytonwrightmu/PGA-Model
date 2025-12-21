"""
ANALYSIS ENGINE - Fragility Analyzer
=====================================
Identifies players likely to struggle in specific conditions.

A "FRAGILE" player is someone who:
❌ Has poor course fit
❌ Has high uncertainty (volatile)
❌ Is low talent
❌ Has elevated miss-cut probability (simple proxy)

USAGE:
    from src.analysis.fragility_analyzer import find_fragile_players

    fades = find_fragile_players(df, "accuracy_premium", top_n=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _minmax(series: pd.Series, default: float = 0.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(default, index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


@dataclass
class FragilityConfig:
    max_course_fit: float = -0.10
    min_miss_cut_pct: float = 35.0
    base_miss_cut_rate: float = 40.0  # baseline, later can vary by event difficulty


class FragilityAnalyzer:
    """Identifies players likely to struggle."""

    def __init__(self, cfg: FragilityConfig = FragilityConfig()):
        self.cfg = cfg

    def estimate_miss_cut_probability(self, df: pd.DataFrame, fit_col: str) -> pd.DataFrame:
        """
        Simple, explainable miss-cut proxy.
        Later you can replace this with a calibrated model using historical event data.
        """
        out = df.copy()

        fit = pd.to_numeric(out[fit_col], errors="coerce").fillna(0.0)
        tal = pd.to_numeric(out["overall_talent"], errors="coerce").fillna(0.0)

        # If uncertainty exists, volatility raises miss-cut chance.
        if "overall_uncertainty" in out.columns:
            unc = pd.to_numeric(out["overall_uncertainty"], errors="coerce")
            unc = unc.fillna(unc.median())
            unc_factor = (unc / max(unc.mean(), 1e-9))
        else:
            unc_factor = pd.Series(1.0, index=out.index, dtype=float)

        base = float(self.cfg.base_miss_cut_rate)

        # These coefficients are intentionally conservative.
        # Poor fit increases risk, low talent increases risk, high uncertainty increases risk.
        miss_cut = (
            base
            + (-15.0 * fit)          # bad fit -> higher %
            + (-10.0 * tal)          # low talent -> higher %
            + (8.0 * (unc_factor - 1.0))  # volatility penalty
        )

        out["miss_cut_probability"] = np.clip(miss_cut, 5.0, 85.0)
        return out

    def calculate_fragility_score(self, df: pd.DataFrame, fit_col: str) -> pd.DataFrame:
        """
        Fragility score in [0,1] ish, higher means more likely to struggle.
        """
        out = df.copy()

        fit = pd.to_numeric(out[fit_col], errors="coerce")
        talent = pd.to_numeric(out["overall_talent"], errors="coerce")

        poor_fit = 1.0 - _minmax(fit.fillna(fit.median()))
        low_talent = 1.0 - _minmax(talent.fillna(talent.median()))

        if "overall_uncertainty" in out.columns:
            unc = pd.to_numeric(out["overall_uncertainty"], errors="coerce")
            high_var = _minmax(unc.fillna(unc.median()))
        else:
            high_var = pd.Series(0.5, index=out.index, dtype=float)

        out["fragility_score"] = (
            0.45 * poor_fit +
            0.30 * high_var +
            0.25 * low_talent
        )

        return out

    def find_fragile_players(self, df: pd.DataFrame, course_archetype: str, top_n: int = 10) -> pd.DataFrame:
        fit_col = f"fit_{course_archetype}"
        if fit_col not in df.columns:
            raise ValueError(f"Course fit column {fit_col} not found")

        needed = ["PLAYER", "overall_talent"]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        out = self.estimate_miss_cut_probability(df, fit_col)
        out = self.calculate_fragility_score(out, fit_col)

        # Candidate filter: poor fit OR high miss-cut probability
        candidates = out[
            (pd.to_numeric(out[fit_col], errors="coerce") <= self.cfg.max_course_fit) |
            (pd.to_numeric(out["miss_cut_probability"], errors="coerce") >= self.cfg.min_miss_cut_pct)
        ].copy()

        fragile = candidates.nlargest(top_n, "fragility_score").copy()

        fragile["risk_level"] = fragile["fragility_score"].apply(
            lambda x: "EXTREME" if x >= 0.75 else "HIGH" if x >= 0.60 else "MODERATE"
        )

        fragile["fragility_reason"] = fragile.apply(
            lambda r: self._reason(r, fit_col),
            axis=1
        )

        cols = [
            "PLAYER",
            "overall_talent",
            fit_col,
            "miss_cut_probability",
            "fragility_score",
            "risk_level",
            "fragility_reason",
        ]
        if "talent_tier" in fragile.columns:
            cols.insert(1, "talent_tier")
        if "overall_uncertainty" in fragile.columns:
            cols.append("overall_uncertainty")

        logger.info("✓ Found %s fragile players", len(fragile))
        return fragile[cols].reset_index(drop=True)

    def _reason(self, row: pd.Series, fit_col: str) -> str:
        reasons = []

        fit = row.get(fit_col, np.nan)
        mcp = row.get("miss_cut_probability", np.nan)
        tal = row.get("overall_talent", np.nan)

        if not pd.isna(fit) and fit <= -0.30:
            reasons.append(f"Bad fit {fit:+.2f}")
        elif not pd.isna(fit) and fit <= -0.10:
            reasons.append(f"Poor fit {fit:+.2f}")

        if not pd.isna(mcp) and mcp >= 55:
            reasons.append(f"Miss-cut risk {mcp:.0f}%")
        elif not pd.isna(mcp) and mcp >= 40:
            reasons.append(f"Elevated miss-cut {mcp:.0f}%")

        if "overall_uncertainty" in row.index and not pd.isna(row["overall_uncertainty"]):
            if row["overall_uncertainty"] >= 0.60:
                reasons.append("High variance")

        if not pd.isna(tal) and tal <= -0.15:
            reasons.append("Low talent")

        return " | ".join(reasons) if reasons else "Multiple risk flags"


def find_fragile_players(
    df: pd.DataFrame,
    course_archetype: str,
    top_n: int = 10,
    max_course_fit: float = -0.10,
    min_miss_cut_pct: float = 35.0,
) -> pd.DataFrame:
    cfg = FragilityConfig(
        max_course_fit=max_course_fit,
        min_miss_cut_pct=min_miss_cut_pct,
    )
    return FragilityAnalyzer(cfg).find_fragile_players(df, course_archetype, top_n)
