"""
DATA ENGINE - Validate Data
============================
Quality checks BEFORE we process data.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates PGA Tour statistics for quality."""

    def __init__(self):
        self.stat_ranges = {
            "sg_ott": (-1.5, 2.0),
            "sg_app": (-1.5, 2.5),
            "sg_arg": (-1.0, 1.5),
            "sg_putt": (-1.5, 2.0),
            "sg_total": (-3.0, 4.0),
        }

        self.min_rounds = 10

        self.expected_correlations = {
            ("sg_ott", "sg_app"): (0.2, 0.6),
            ("sg_app", "sg_arg"): (0.1, 0.5),
            ("sg_putt", "sg_ott"): (-0.2, 0.2),
        }

    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        missing = {}
        for col in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
            if col not in df.columns:
                missing[col] = "COLUMN MISSING"
                continue
            pct_missing = df[col].isna().sum() / max(len(df), 1) * 100
            if pct_missing > 0:
                missing[col] = f"{pct_missing:.1f}% missing"
        return missing

    def check_stat_ranges(self, df: pd.DataFrame) -> Dict:
        issues = {}
        for stat, (min_val, max_val) in self.stat_ranges.items():
            if stat not in df.columns:
                continue
            outliers = df[(df[stat] < min_val) | (df[stat] > max_val)]
            if len(outliers) > 0:
                issues[stat] = {
                    "n_outliers": int(len(outliers)),
                    "expected_range": (min_val, max_val),
                    "example_values": outliers[stat].dropna().head(8).tolist(),
                }
        return issues

    def check_round_counts(self, df: pd.DataFrame) -> Dict:
        round_cols = [c for c in df.columns if c.endswith("_rounds")]
        if not round_cols:
            return {"warning": "No round count columns found"}

        if "total_rounds" in df.columns:
            low = df[df["total_rounds"] < self.min_rounds]
            return {
                "players_with_low_rounds": int(len(low)),
                "min_rounds_threshold": self.min_rounds,
                "median_rounds": float(df["total_rounds"].median()),
            }
        return {}

    def check_correlations(self, df: pd.DataFrame) -> Dict:
        issues = {}
        for (a, b), (lo, hi) in self.expected_correlations.items():
            if a not in df.columns or b not in df.columns:
                continue
            sub = df[[a, b]].dropna()
            if len(sub) < 30:
                continue
            corr = sub[a].corr(sub[b])
            if corr < lo or corr > hi:
                issues[f"{a}_vs_{b}"] = {
                    "actual_correlation": round(float(corr), 3),
                    "expected_range": (lo, hi),
                }
        return issues

    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        if "PLAYER_ID" not in df.columns:
            return {"error": "No PLAYER_ID column"}
        dupes = df[df.duplicated(subset=["PLAYER_ID"], keep=False)]
        return {
            "has_duplicates": len(dupes) > 0,
            "n_duplicates": int(len(dupes)),
            "duplicate_ids": dupes["PLAYER_ID"].dropna().astype(str).head(20).tolist(),
        }

    def validate_all(self, df: pd.DataFrame) -> Dict:
        logger.info("Running data validation...")

        report = {"passed": True, "errors": [], "warnings": [], "checks": {}}

        missing = self.check_missing_values(df)
        report["checks"]["missing_values"] = missing
        if missing:
            report["warnings"].append(f"Missing values detected in {len(missing)} columns")

        range_issues = self.check_stat_ranges(df)
        report["checks"]["stat_ranges"] = range_issues
        if range_issues:
            report["warnings"].append(f"Outliers detected in {len(range_issues)} stats")

        rounds = self.check_round_counts(df)
        report["checks"]["round_counts"] = rounds
        if rounds.get("players_with_low_rounds", 0) > 10:
            report["warnings"].append(
                f"{rounds['players_with_low_rounds']} players have < {self.min_rounds} rounds"
            )

        corrs = self.check_correlations(df)
        report["checks"]["correlations"] = corrs
        if corrs:
            report["warnings"].append(f"Unexpected correlations: {len(corrs)}")

        dupes = self.check_duplicates(df)
        report["checks"]["duplicates"] = dupes
        if dupes.get("has_duplicates"):
            report["errors"].append(f"Duplicate PLAYER_ID values found: {dupes['n_duplicates']}")
            report["passed"] = False

        if report["errors"]:
            logger.error("✗ Validation FAILED: %s errors", len(report["errors"]))
            report["passed"] = False
        elif report["warnings"]:
            logger.warning("⚠ Validation passed with %s warnings", len(report["warnings"]))
        else:
            logger.info("✓ Validation PASSED")

        return report


def validate_player_data(df: pd.DataFrame) -> Dict:
    validator = DataValidator()
    return validator.validate_all(df)


def print_validation_report(report: Dict) -> None:
    print("\n" + "=" * 80)
    print("DATA VALIDATION REPORT")
    print("=" * 80)

    print("\n✓ VALIDATION PASSED" if report["passed"] else "\n✗ VALIDATION FAILED")

    if report["errors"]:
        print(f"\nERRORS ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  - {e}")

    if report["warnings"]:
        print(f"\nWARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  - {w}")

    print("\n" + "-" * 80)
    print("CHECKS")
    print("-" * 80)

    for name, payload in report["checks"].items():
        print(f"\n{name.upper()}:")
        if isinstance(payload, dict):
            for k, v in payload.items():
                print(f"  {k}: {v}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_engine.load_raw_stats import load_all_player_stats

    df = load_all_player_stats(year=2025)
    rep = validate_player_data(df)
    print_validation_report(rep)
