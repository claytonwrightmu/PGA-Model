from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from src.analysis.condition_matcher import Conditions


@dataclass
class TournamentContext:
    name: str
    year: int
    archetype_mix: Dict[str, float]
    conditions: Conditions = field(default_factory=Conditions)
    field_player_ids: Optional[List[int]] = None


def get_tournament_context(name: str, year: int = 2025) -> TournamentContext:
    """
    Hardcode a few templates for now.
    Later: load from JSON in data/processed/tournaments/.
    """
    key = name.strip().lower()

    presets: Dict[str, TournamentContext] = {
        "template_bomber": TournamentContext(
            name="template_bomber",
            year=year,
            archetype_mix={"bomber_paradise": 1.0},
            conditions=Conditions(wind_mph=10, rough="medium", course_difficulty="neutral"),
        ),
        "template_accuracy": TournamentContext(
            name="template_accuracy",
            year=year,
            archetype_mix={"accuracy_premium": 1.0},
            conditions=Conditions(wind_mph=12, rough="heavy", course_difficulty="hard"),
        ),
        "template_mix": TournamentContext(
            name="template_mix",
            year=year,
            archetype_mix={"bomber_paradise": 0.6, "target_golf": 0.4},
            conditions=Conditions(wind_mph=15, rough="heavy", course_difficulty="hard"),
        ),
    }

    if key not in presets:
        raise ValueError(f"Unknown tournament context '{name}'. Available: {list(presets.keys())}")

    return presets[key]
