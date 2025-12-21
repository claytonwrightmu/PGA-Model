"""
DATA ENGINE - Calculate Talent
===============================
Bayesian estimation of TRUE player talent.

WHY BAYESIAN?
- Players with few rounds get shrunk toward average
- Players with many rounds stay close to observed stats
- Helps separate luck vs skill
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BayesianTalentEstimator:
    """
    Estimates true player talent using Bayesian shrinkage.

    Note: This is a pragmatic shrinkage model, not a full hierarchical Bayes model.
    It's designed to be stable + useful for betting/DFS workflows.
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_std: float = 0.5,
        base_uncertainty: float = 0.30,
    ):
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)
        self.base_uncertainty = float(base_uncertainty)

    def estimate_single_skill(
        self,
        observed_avg: float,
        n_rounds: float,
        round_variance: float = 6.25,
    ) -> Tuple[float, float]:
        """
        Posterior mean + uncertainty under a Normal prior and Normal likelihood.

        measurement_variance = round_variance / n_rounds
        shrinkage = prior_variance / (prior_variance + measurement_variance)
        """
        if pd.isna(observed_avg) or pd.isna(n_rounds) or n_rounds <= 0:
            return self.prior_mean, self.prior_std

        prior_var = self.prior_std ** 2
        meas_var = float(round_variance) / float(n_rounds)

        shrinkage = prior_var / (prior_var + meas_var)

        talent = shrinkage * float(observed_avg) + (1.0 - shrinkage) * self.prior_mean

        posterior_var = shrinkage * meas_var
        uncertainty = float(np.sqrt(posterior_var + self.base_uncertainty**2))

        return float(talent), float(uncertainty)

    def estimate_all_skills(self, df: pd.DataFrame, skills: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        if skills is None:
            skills = {
                "sg_ott": "sg_ott",
                "sg_app": "sg_app",
                "sg_arg": "sg_arg",
                "sg_putt": "sg_putt",
            }

        out = df.copy()

        for skill_name, col_name in skills.items():
            if col_name not in out.columns:
                logger.warning("Column %s not found, skipping %s", col_name, skill_name)
                continue

            rounds_col = f"{skill_name}_rounds"
            if rounds_col not in out.columns:
                rounds_col = "total_rounds"

            if rounds_col not in out.columns:
                logger.warning("No rounds column for %s (expected %s or total_rounds)", skill_name, f"{skill_name}_rounds")
                continue

            talents = []
            uncs = []
            shrink_pcts = []

            obs_series = pd.to_numeric(out[col_name], errors="coerce")
            rnd_series = pd.to_numeric(out[rounds_col], errors="coerce")

            for obs, rnd in zip(obs_series.values, rnd_series.values):
                t, u = self.estimate_single_skill(obs, rnd)
                talents.append(t)
                uncs.append(u)

                if pd.isna(obs):
                    shrink_pcts.append(np.nan)
                else:
                    denom = abs(float(obs) - self.prior_mean) + 1e-10
                    shrink_pcts.append(abs(t - float(obs)) / denom)

            out[f"{skill_name}_talent"] = talents
            out[f"{skill_name}_uncertainty"] = uncs
            out[f"{skill_name}_shrinkage_pct"] = shrink_pcts

        return out

    def calculate_overall_talent(self, df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        if weights is None:
            weights = {
                "sg_ott_talent": 0.25,
                "sg_app_talent": 0.25,
                "sg_arg_talent": 0.25,
                "sg_putt_talent": 0.25,
            }

        out = df.copy()
        talent_cols = [c for c in weights.keys() if c in out.columns]
        if not talent_cols:
            logger.error("No talent columns found to compute overall_talent")
            return out

        total_w = sum(weights[c] for c in talent_cols)
        w = {c: weights[c] / total_w for c in talent_cols}

        out["overall_talent"] = 0.0
        for col, wt in w.items():
            out["overall_talent"] += pd.to_numeric(out[col], errors="coerce").fillna(0.0) * float(wt)

        unc_cols = [c.replace("_talent", "_uncertainty") for c in talent_cols]
        if all(c in out.columns for c in unc_cols):
            var_sum = 0.0
            for tcol, ucol in zip(talent_cols, unc_cols):
                var_sum += (pd.to_numeric(out[ucol], errors="coerce").fillna(0.0) ** 2) * (weights[tcol] ** 2)
            out["overall_uncertainty"] = np.sqrt(var_sum)

        return out

    def assign_tiers(self, df: pd.DataFrame, talent_col: str = "overall_talent") -> pd.DataFrame:
        out = df.copy()
        if talent_col not in out.columns:
            return out

        out["talent_percentile"] = out[talent_col].rank(pct=True) * 100.0

        def tier(p: float) -> str:
            if pd.isna(p):
                return "Unranked"
            if p >= 95:
                return "S"
            if p >= 85:
                return "A"
            if p >= 70:
                return "B"
            if p >= 50:
                return "C"
            return "D"

        out["talent_tier"] = out["talent_percentile"].apply(tier)
        return out


def estimate_all_talents(df: pd.DataFrame, prior_mean: float = 0.0, prior_std: float = 0.5) -> pd.DataFrame:
    logger.info("Estimating player talents...")

    est = BayesianTalentEstimator(prior_mean=prior_mean, prior_std=prior_std)

    out = est.estimate_all_skills(df)
    out = est.calculate_overall_talent(out)
    out = est.assign_tiers(out)

    tier_counts = out["talent_tier"].value_counts().to_dict() if "talent_tier" in out.columns else {}
    logger.info("✓ Estimated talents for %s players", len(out))
    if tier_counts:
        logger.info("Tier distribution: %s", tier_counts)

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_engine.load_raw_stats import load_all_player_stats

    df0 = load_all_player_stats(year=2025)
    df1 = estimate_all_talents(df0)

    cols = ["PLAYER", "overall_talent", "talent_tier", "sg_ott_talent", "sg_app_talent"]
    cols = [c for c in cols if c in df1.columns]
    print(df1.sort_values("overall_talent", ascending=False)[cols].head(10))
