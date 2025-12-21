"""
DATA ENGINE - Load Raw Stats
=============================
Loads PGA Tour statistics from Excel files and creates unified player database.

Key upgrades vs earlier version:
- Robust PLAYER_ID -> PLAYER mapping by scanning ALL raw Excel files (fixes missing names).
- Smarter detection of value/round columns (handles messy PGA export column names).
- Faster reads (reads header first, then only needed columns where possible).
- Better rankings discovery + flexible rank column detection.
- Clearer logging and safer merges.

USAGE:
    from src.data_engine.load_raw_stats import load_all_player_stats
    df = load_all_player_stats(year=2025)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return (
        str(s).lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("'", " ")
        .replace("%", " percent ")
        .replace(":", " ")
        .strip()
    )


def _best_col_match(cols: List[str], candidates: List[str]) -> Optional[str]:
    """
    Return the first column in cols that matches any normalized candidate substring.
    """
    cols_norm = {c: _norm(c) for c in cols}
    cand_norm = [_norm(x) for x in candidates]

    for c, cn in cols_norm.items():
        for pat in cand_norm:
            if pat in cn:
                return c
    return None


def _coerce_player_id(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # Some files may include floats like 1234.0
    return s.round().astype("Int64")


class PGADataLoader:
    """Loads PGA Tour statistics from Excel files."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)

        self.stat_keys = [
            "sg_ott",
            "sg_app",
            "sg_arg",
            "sg_putt",
            "sg_t2g",
            "driving_distance",
            "driving_accuracy",
            "gir",
            "scrambling",
            "sand_save",
        ]

        # File name patterns (case-insensitive, normalized)
        self.file_patterns: Dict[str, List[str]] = {
            "sg_ott": ["strokes gained off the tee", "sg off the tee", "sg ott", "sg_ott"],
            "sg_app": ["strokes gained approach", "sg approach", "sg app", "sg_app"],
            "sg_arg": ["strokes gained around the green", "sg around", "sg arg", "sg_arg"],
            "sg_putt": ["strokes gained putting", "sg putting", "sg putt", "sg_putt"],
            "sg_t2g": ["strokes gained tee to green", "sg tee to green", "sg t2g", "sg_t2g"],
            "driving_distance": ["driving distance", "distance leaders", "ball speed leaders"],
            "driving_accuracy": ["driving accuracy"],
            "gir": ["gir percentage", "greens in regulation", "gir %", "gir percent"],
            "scrambling": ["scrambling leaders", "scrambling"],
            "sand_save": ["sand save", "sand saves", "sand save %", "sand save percent"],
        }

        # Candidate column names for value/rounds across various PGA exports
        self.value_col_candidates = [
            "AVG",
            "AVERAGE",
            "VALUE",
            "SG",
            "STROKES GAINED",
            "PER ROUND",
            "MEASURE",
            "STAT",
            "PERCENTAGE",
            "%",
        ]
        self.rounds_col_candidates = [
            "MEASURED ROUNDS",
            "ROUNDS",
            "RNDS",
            "RND",
            "TOTAL ROUNDS",
        ]

    def _find_file(self, year: int, stat_key: str) -> Optional[Path]:
        # Year prefix convention: 23' / 24' etc; 2025 may or may not have prefix in your setup
        year_tag = ""
        if year == 2023:
            year_tag = "23"
        elif year == 2024:
            year_tag = "24"
        elif year == 2025:
            year_tag = ""  # allow both prefixed and non-prefixed

        excel_files = list(self.data_path.glob("*.xlsx"))
        patterns = self.file_patterns.get(stat_key, [stat_key])
        patterns_norm = [_norm(p) for p in patterns]

        best: Optional[Path] = None

        for file in excel_files:
            fn = _norm(file.name)

            # If 2023/2024, require year tag
            if year_tag and year_tag not in fn:
                continue

            # If 2025, allow either no tag or "25"
            if year == 2025:
                # do not exclude on year tag
                pass

            for p in patterns_norm:
                if p in fn:
                    best = file
                    break

            if best is not None:
                break

        return best

    def _read_excel_columns(self, file_path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read Excel file. If usecols is provided, reads only those columns when possible.
        Falls back to full read if engine fails.
        """
        try:
            if usecols:
                return pd.read_excel(file_path, usecols=usecols)
            return pd.read_excel(file_path)
        except Exception:
            # Fallback
            return pd.read_excel(file_path)

    def _detect_value_and_rounds_cols(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        cols = list(df.columns)

        # Direct hits first
        value_col = None
        for c in ["AVG", "AVERAGE", "VALUE"]:
            if c in df.columns:
                value_col = c
                break

        rounds_col = None
        for c in ["MEASURED ROUNDS", "ROUNDS", "RNDS"]:
            if c in df.columns:
                rounds_col = c
                break

        # Smarter fuzzy detection
        if value_col is None:
            value_col = _best_col_match(cols, self.value_col_candidates)

        if rounds_col is None:
            rounds_col = _best_col_match(cols, self.rounds_col_candidates)

        # Avoid selecting PLAYER or PLAYER_ID accidentally
        if value_col in ("PLAYER", "PLAYER_ID"):
            value_col = None
        if rounds_col in ("PLAYER", "PLAYER_ID"):
            rounds_col = None

        return value_col, rounds_col

    def load_stat(self, year: int, stat_key: str) -> Optional[pd.DataFrame]:
        file_path = self._find_file(year, stat_key)

        if not file_path:
            logger.warning("File not found for %s (%s)", stat_key, year)
            return None

        # Read full, then detect columns (more robust than trying to guess usecols up front)
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            return None

        if "PLAYER_ID" not in df.columns:
            logger.error("Missing PLAYER_ID in %s", file_path.name)
            return None

        value_col, rounds_col = self._detect_value_and_rounds_cols(df)

        result = df[["PLAYER_ID"]].copy()
        result["PLAYER_ID"] = _coerce_player_id(result["PLAYER_ID"])

        if value_col:
            result[stat_key] = pd.to_numeric(df[value_col], errors="coerce")
        else:
            logger.warning("No value column found in %s for %s", file_path.name, stat_key)

        if rounds_col:
            result[f"{stat_key}_rounds"] = pd.to_numeric(df[rounds_col], errors="coerce")

        # Drop rows with no valid PLAYER_ID
        result = result.dropna(subset=["PLAYER_ID"]).copy()
        return result

    def load_all_stats(self, year: int) -> pd.DataFrame:
        logger.info("Loading stats for %s...", year)

        dfs: List[pd.DataFrame] = []
        for stat_key in self.stat_keys:
            df = self.load_stat(year, stat_key)
            if df is not None:
                dfs.append(df)

        if not dfs:
            raise ValueError(f"No stats loaded for {year}")

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="PLAYER_ID", how="outer")

        merged["year"] = year
        merged = merged.dropna(subset=["PLAYER_ID"]).copy()

        logger.info("Loaded %s players, %s stat files", len(merged), len(dfs))
        return merged

    def load_player_names(self) -> pd.DataFrame:
        """
        Robust PLAYER_ID -> PLAYER mapping by scanning ALL Excel files.
        This fixes missing names that occur when you only use one stat file for names.
        """
        excel_files = list(self.data_path.glob("*.xlsx"))
        if not excel_files:
            raise FileNotFoundError(f"No .xlsx files found in {self.data_path}")

        name_frames: List[pd.DataFrame] = []
        for file in excel_files:
            try:
                tmp = pd.read_excel(file)
            except Exception:
                continue

            if "PLAYER_ID" in tmp.columns and "PLAYER" in tmp.columns:
                chunk = tmp[["PLAYER_ID", "PLAYER"]].copy()
                chunk["PLAYER_ID"] = _coerce_player_id(chunk["PLAYER_ID"])
                chunk["PLAYER"] = chunk["PLAYER"].astype(str).str.strip()
                chunk = chunk.dropna(subset=["PLAYER_ID", "PLAYER"])
                chunk = chunk[chunk["PLAYER"].str.len() > 0]
                name_frames.append(chunk)

        if not name_frames:
            raise FileNotFoundError("Could not find PLAYER_ID + PLAYER columns in any raw stat files")

        names = pd.concat(name_frames, ignore_index=True)
        names = names.drop_duplicates(subset=["PLAYER_ID", "PLAYER"])

        # Prefer the most common name for each ID (handles minor variations)
        names = (
            names.groupby("PLAYER_ID")["PLAYER"]
            .agg(lambda s: s.value_counts().idxmax())
            .reset_index()
        )

        logger.info("✓ Built player name map from raw files (%s unique names)", len(names))
        return names

    def load_world_rankings(self) -> pd.DataFrame:
        """
        Load OWGR-like rankings.
        We look for any *rank*.xlsx in raw folder.
        Flexible column detection: rank might be RANK, OWGR, WORLD_RANK, etc.
        """
        files = sorted(self.data_path.glob("*rank*.xlsx"))
        if not files:
            logger.warning("No ranking files found in %s", self.data_path)
            return pd.DataFrame(columns=["PLAYER_ID", "world_rank"])

        # Prefer newest / most relevant by filename sort
        file = files[-1]

        try:
            df = pd.read_excel(file)
        except Exception as e:
            logger.warning("Could not read ranking file %s: %s", file.name, e)
            return pd.DataFrame(columns=["PLAYER_ID", "world_rank"])

        if "PLAYER_ID" not in df.columns:
            logger.warning("Ranking file %s missing PLAYER_ID", file.name)
            return pd.DataFrame(columns=["PLAYER_ID", "world_rank"])

        rank_col = None
        for c in ["RANK", "WORLD_RANK", "OWGR", "RK", "POSITION"]:
            if c in df.columns:
                rank_col = c
                break

        if rank_col is None:
            rank_col = _best_col_match(list(df.columns), ["world rank", "rank", "owgr", "position"])

        if rank_col is None:
            logger.warning("Ranking file %s has no recognizable rank column", file.name)
            return pd.DataFrame(columns=["PLAYER_ID", "world_rank"])

        out = df[["PLAYER_ID", rank_col]].copy()
        out["PLAYER_ID"] = _coerce_player_id(out["PLAYER_ID"])
        out["world_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
        out = out.dropna(subset=["PLAYER_ID"]).copy()
        out = out[["PLAYER_ID", "world_rank"]].drop_duplicates(subset=["PLAYER_ID"])

        logger.info("✓ Loaded world rankings from %s (%s rows)", file.name, len(out))
        return out


def load_all_player_stats(
    year: int = 2025,
    data_path: Optional[Path] = None,
    include_history: bool = True,
) -> pd.DataFrame:
    """
    MAIN FUNCTION: Load complete player database for a year (optionally add historical career features).
    """
    if data_path is None:
        from config import DATA_RAW
        data_path = DATA_RAW

    loader = PGADataLoader(data_path)

    # Current season
    current = loader.load_all_stats(year)

    # Add player names (robust map)
    names = loader.load_player_names()
    current = current.merge(names, on="PLAYER_ID", how="left")

    # Add world rankings
    rankings = loader.load_world_rankings()
    current = current.merge(rankings, on="PLAYER_ID", how="left")

    # Include historical averages for Bayesian priors/features
    if include_history and year >= 2024:
        hist_years = [y for y in [2023, 2024] if y < year]
        history_dfs: List[pd.DataFrame] = []

        for hy in hist_years:
            try:
                hist = loader.load_all_stats(hy)
                history_dfs.append(hist)
            except Exception as e:
                logger.warning("Could not load %s: %s", hy, e)

        if history_dfs:
            history = pd.concat(history_dfs, ignore_index=True)
            core = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

            career = history.groupby("PLAYER_ID")[core].agg(["mean", "std", "count"])
            career.columns = [f"{stat}_{agg}_career" for stat, agg in career.columns]
            career = career.reset_index()

            current = current.merge(career, on="PLAYER_ID", how="left")

    # Total rounds measured (if any)
    round_cols = [c for c in current.columns if c.endswith("_rounds")]
    if round_cols:
        current["total_rounds"] = current[round_cols].sum(axis=1, skipna=True)

    logger.info("✓ Master database: %s players, %s columns", len(current), len(current.columns))
    return current


if __name__ == "__main__":
    df = load_all_player_stats(year=2025, data_path=Path("data/raw"), include_history=True)
    print(df.head())
