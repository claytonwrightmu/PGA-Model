"""
ANALYSIS ENGINE - Sleeper Detector
===================================
Finds UNDERVALUED players who are primed to outperform.

A "SLEEPER" is a player who:
✓ Has strong talent (Bayesian estimate)
✓ Has excellent course fit
✓ Is NOT getting public attention (optional world_rank filter)

USAGE:
    from src.analysis.sleeper_detector import find_sleepers

    sleepers = find_sleepers(df, course_archetype="bomber_paradise", top_n=10)
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
class SleeperConfig:
    min_course_fit: float = 0.20
    min_talent_percentile: float = 60.0
    max_public_rank: int = 50  # world_rank must be > this to count as "under the radar"
    exclude_tiers: tuple[str, ...] = ("S", "A")


class SleeperDetector:
    """Identifies undervalued players with high upside."""

    def __init__(self, cfg: SleeperConfig = SleeperConfig()):
        self.cfg = cfg

    def _public_perception(self, df: pd.DataFrame) -> pd.Series:
        """
        Perception score in [0,1], where 1 = highly known, 0 = unknown.
        Uses world_rank if present; otherwise returns neutral.
        """
        if "world_rank" not in df.columns:
            return pd.Series(0.5, index=df.index, dtype=float)

        wr = pd.to_numeric(df["world_rank"], errors="coerce")
        # Lower world_rank means more famous, so perception should be higher.
        # Convert to 0-1 then flip appropriately.
        # If world_rank has NaNs, keep neutral perception for those.
        wr_norm = _minmax(wr.fillna(wr.median()))
        perception = 1.0 - wr_norm  # low rank -> high perception
        perception = perception.where(~wr.isna(), 0.5)
        return perception

    def calculate_value_score(self, df: pd.DataFrame, fit_col: str) -> pd.DataFrame:
        """
        Value score: rewards talent + fit (+ optional form), penalizes public perception.
        Higher = more undervalued.
        """
        out = df.copy()

        # Required base columns
        for col in ["overall_talent", "talent_percentile", "PLAYER"]:
            if col not in out.columns:
                raise ValueError(f"Missing required column: {col}")

        if fit_col not in out.columns:
            raise ValueError(f"Missing required fit column: {fit_col}")

        talent_norm = _minmax(out["overall_talent"])
        fit_norm = _minmax(out[fit_col])

        # Optional: form, if you later add a form engine.
        if "form" in out.columns:
            form_norm = _minmax(out["form"])
        else:
            form_norm = pd.Series(0.5, index=out.index, dtype=float)

        perception = self._public_perception(out)

        # Penalize uncertainty a bit (sleepers should be upside but not pure chaos)
        if "overall_uncertainty" in out.columns:
            unc_norm = _minmax(out["overall_uncertainty"])
        else:
            unc_norm = pd.Series(0.5, index=out.index, dtype=float)

        # Value score: core is talent + fit. Form is smaller weight. Uncertainty penalized.
        raw = (
            0.40 * talent_norm +
            0.40 * fit_norm +
            0.15 * form_norm -
            0.05 * unc_norm
        )

        # Divide by perception (more famous -> lower value)
        out["value_score"] = raw / (perception + 0.10)

        return out

    def find_sleepers(self, df: pd.DataFrame, course_archetype: str, top_n: int = 10) -> pd.DataFrame:
        fit_col = f"fit_{course_archetype}"
        out = self.calculate_value_score(df, fit_col)

        # Base filters
        candidates = out[
            (pd.to_numeric(out["talent_percentile"], errors="coerce") >= self.cfg.min_talent_percentile) &
            (pd.to_numeric(out[fit_col], errors="coerce") >= self.cfg.min_course_fit)
        ].copy()

        # Public attention filter
        if "world_rank" in candidates.columns:
            wr = pd.to_numeric(candidates["world_rank"], errors="coerce")
            candidates = candidates[(wr > self.cfg.max_public_rank) | (wr.isna())]

        # Exclude top tiers (not sleepers, they're favorites)
        if "talent_tier" in candidates.columns and self.cfg.exclude_tiers:
            candidates = candidates[~candidates["talent_tier"].isin(self.cfg.exclude_tiers)]

        sleepers = candidates.nlargest(top_n, "value_score").copy()

        sleepers["sleeper_reason"] = sleepers.apply(
            lambda r: self._reason(r, fit_col),
            axis=1
        )

        cols = [
            "PLAYER",
            "talent_tier",
            "overall_talent",
            "talent_percentile",
            fit_col,
            "value_score",
            "sleeper_reason",
        ]
        if "world_rank" in sleepers.columns:
            cols.append("world_rank")
        if "overall_uncertainty" in sleepers.columns:
            cols.append("overall_uncertainty")

        logger.info("✓ Found %s sleeper candidates", len(sleepers))
        return sleepers[cols].reset_index(drop=True)

    def _reason(self, row: pd.Series, fit_col: str) -> str:
        reasons = []

        tal = row.get("overall_talent", np.nan)
        fit = row.get(fit_col, np.nan)
        tier = row.get("talent_tier", "")

        if not pd.isna(tal) and tal >= 0.25:
            reasons.append(f"Talent {tal:+.2f} SG/rd")
        if not pd.isna(fit) and fit >= 0.50:
            reasons.append(f"Elite fit {fit:+.2f}")
        elif not pd.isna(fit) and fit >= 0.30:
            reasons.append(f"Strong fit {fit:+.2f}")

        if "world_rank" in row.index and not pd.isna(row["world_rank"]):
            if row["world_rank"] > 75:
                reasons.append(f"Low hype (WR {int(row['world_rank'])})")

        if tier in ("B", "C", "D"):
            reasons.append(f"{tier}-tier upside")

        return " | ".join(reasons) if reasons else "Value candidate"


def find_sleepers(
    df: pd.DataFrame,
    course_archetype: str,
    top_n: int = 10,
    min_course_fit: float = 0.20,
    min_talent_percentile: float = 60.0,
    max_public_rank: int = 50,
) -> pd.DataFrame:
    cfg = SleeperConfig(
        min_course_fit=min_course_fit,
        min_talent_percentile=min_talent_percentile,
        max_public_rank=max_public_rank,
    )
    return SleeperDetector(cfg).find_sleepers(df, course_archetype, top_n)
