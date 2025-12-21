"""
Configuration - All Settings in ONE Place
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Data paths
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_HISTORICAL = ROOT_DIR / "data" / "historical"
DATA_VALIDATION = ROOT_DIR / "data" / "validation"

# Output paths
OUTPUT_REPORTS = ROOT_DIR / "outputs" / "reports"
OUTPUT_PREDICTIONS = ROOT_DIR / "outputs" / "predictions"
OUTPUT_CHARTS = ROOT_DIR / "outputs" / "charts"

# Model settings
PRIOR_MEAN = 0.0
PRIOR_STD = 0.5
N_SIMULATIONS = 10000

# Course archetypes (9-type system)
COURSE_ARCHETYPES = {
    "bomber_paradise": {
        "name": "Bomber Paradise",
        "description": "Long, wide courses where distance dominates",
        "weights": {"sg_ott": 0.45, "sg_app": 0.30, "sg_arg": 0.10, "sg_putt": 0.15},
        "examples": ["Kapalua", "WM Phoenix Open"],
    },
    "classic_ballstriking": {
        "name": "Classic Ballstriking",
        "description": "Premium on iron play and ball-striking",
        "weights": {"sg_ott": 0.20, "sg_app": 0.45, "sg_arg": 0.20, "sg_putt": 0.15},
        "examples": ["Riviera", "Memorial", "Genesis Invitational"],
    },
    "accuracy_premium": {
        "name": "Accuracy Premium",
        "description": "Tight, tree-lined with heavy rough",
        "weights": {"sg_ott": 0.20, "sg_app": 0.35, "sg_arg": 0.25, "sg_putt": 0.20},
        "examples": ["Harbour Town", "Colonial"],
    },
    "target_golf": {
        "name": "Target Golf",
        "description": "Strategic shot-making with hazards",
        "weights": {"sg_ott": 0.25, "sg_app": 0.35, "sg_arg": 0.25, "sg_putt": 0.15},
        "examples": ["TPC Sawgrass", "Bay Hill"],
    },
    "major_championship": {
        "name": "Major Championship",
        "description": "Severe setup with major pressure",
        "weights": {"sg_ott": 0.25, "sg_app": 0.35, "sg_arg": 0.25, "sg_putt": 0.15},
        "examples": ["Augusta", "US Open venues"],
    },
    "putting_contest": {
        "name": "Putting Contest",
        "description": "Easy greens or birdie fests where putting matters more",
        "weights": {"sg_ott": 0.15, "sg_app": 0.25, "sg_arg": 0.10, "sg_putt": 0.50},
        "examples": ["TPC Summerlin", "Detroit GC"],
    },
    "short_game_test": {
        "name": "Short Game Test",
        "description": "Missed greens are common, scrambling separates",
        "weights": {"sg_ott": 0.15, "sg_app": 0.25, "sg_arg": 0.45, "sg_putt": 0.15},
        "examples": ["Pebble Beach", "Muirfield Village (windy setups)"],
    },
    "wind_specialist": {
        "name": "Wind Specialist",
        "description": "Coastal or exposed venues where control matters",
        "weights": {"sg_ott": 0.20, "sg_app": 0.40, "sg_arg": 0.25, "sg_putt": 0.15},
        "examples": ["The Open venues", "Waialae (trade winds)"],
    },
    "rough_penalty": {
        "name": "Rough Penalty",
        "description": "Heavy rough punishes misses, approach from rough is brutal",
        "weights": {"sg_ott": 0.25, "sg_app": 0.40, "sg_arg": 0.20, "sg_putt": 0.15},
        "examples": ["Torrey Pines", "US Open style venues"],
    },
}


def ensure_dirs() -> None:
    for p in (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_HISTORICAL,
        DATA_VALIDATION,
        OUTPUT_REPORTS,
        OUTPUT_PREDICTIONS,
        OUTPUT_CHARTS,
    ):
        p.mkdir(parents=True, exist_ok=True)
