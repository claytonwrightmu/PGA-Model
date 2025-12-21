from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import DATA_HISTORICAL
from src.utils.helpers import minmax

logger = logging.getLogger(__name__)


REQUIRED_COLS = ["event_id", "PLAYER_ID", "finish_position"]


def load_historical_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Historical results missing columns: {missing}")
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    return df


def evaluate_signal(df: pd.DataFrame, signal_col: str) -> Dict:
    """
    Return simple diagnostics: correlation with finish, and top decile hit rate.
    Lower finish_position is better.
    """
    if signal_col not in df.columns:
        return {"error": f"missing signal col {signal_col}"}

    sub = df[[signal_col, "finish_position"]].dropna()
    if len(sub) < 50:
        return {"warning": "not enough rows", "n": int(len(sub))}

    corr = sub[signal_col].corr(-sub["finish_position"])  # higher better vs lower finish
    score = minmax(sub[signal_col])
    sub = sub.assign(_score=score.values)

    top_decile = sub[sub["_score"] >= 0.9]
    hit_rate_top20 = (top_decile["finish_position"] <= 20).mean()

    return {
        "n": int(len(sub)),
        "corr_with_-finish": float(corr),
        "top_decile_n": int(len(top_decile)),
        "top_decile_top20_rate": float(hit_rate_top20),
    }


def validate_fits(master_df: pd.DataFrame, results_df: pd.DataFrame, archetype_key: str) -> Dict:
    """
    master_df must include PLAYER_ID, overall_talent, and fit_<archetype_key>
    results_df includes event results
    archetype_key is what the event is labeled as, for now we pass it manually.
    """
    fit_col = f"fit_{archetype_key}"
    joined = results_df.merge(master_df, on="PLAYER_ID", how="left")

    out = {
        "overall_talent": evaluate_signal(joined, "overall_talent"),
        fit_col: evaluate_signal(joined, fit_col),
    }
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # This is a validation utility, you call it after building master_df.
    # Example usage is implemented in validation/backtest_predictions.py
    print("Run validation/backtest_predictions.py instead.")
