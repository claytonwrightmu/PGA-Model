from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def minmax(series: pd.Series, default: float = 0.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(default, index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def pick_first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def stamp(prefix: str) -> str:
    import datetime as _dt
    return f"{prefix}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
