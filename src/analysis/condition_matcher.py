from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Conditions:
    wind_mph: Optional[float] = None
    firmness: Optional[str] = None  # soft, neutral, firm
    rough: Optional[str] = None     # light, medium, heavy
    course_difficulty: Optional[str] = None  # easy, neutral, hard


def apply_condition_modifiers(df: pd.DataFrame, conditions: Conditions) -> pd.DataFrame:
    """
    Adds simple columns that can be used by predictor later.
    This does not overwrite talent or fit, it creates deltas.
    """
    out = df.copy()
    out["cond_fit_delta"] = 0.0

    if conditions.wind_mph is not None:
        # wind helps control players, hurts volatility
        # this is a placeholder signal, later we can learn coefficients from historical
        wind = float(conditions.wind_mph)
        out["cond_fit_delta"] += (wind / 20.0) * 0.05  # up to about +0.05

        if "overall_uncertainty" in out.columns:
            out["cond_fit_delta"] -= (out["overall_uncertainty"].fillna(out["overall_uncertainty"].median()) - 0.4) * 0.05

    if conditions.rough is not None:
        if conditions.rough == "heavy":
            # penalize weak OTT slightly
            if "sg_ott_talent" in out.columns:
                out["cond_fit_delta"] += out["sg_ott_talent"].fillna(0.0) * 0.03
        if conditions.rough == "light":
            out["cond_fit_delta"] += 0.01

    if conditions.course_difficulty is not None:
        # saved for later integration into miss-cut base rate
        out["course_difficulty_flag"] = conditions.course_difficulty

    return out
