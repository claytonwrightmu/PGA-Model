"""
Configuration - All Settings in ONE Place
==========================================
"""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent

# Data paths
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_HISTORICAL = ROOT_DIR / "data" / "historical"

# Output paths
OUTPUT_REPORTS = ROOT_DIR / "outputs" / "reports"
OUTPUT_PREDICTIONS = ROOT_DIR / "outputs" / "predictions"
OUTPUT_CHARTS = ROOT_DIR / "outputs" / "charts"

# Model settings
PRIOR_MEAN = 0.0        # Population mean for Bayesian estimation
PRIOR_STD = 0.5         # Population standard deviation
N_SIMULATIONS = 10000   # Monte Carlo iterations

# Course archetypes (we'll add these next)
COURSE_ARCHETYPES = {
    "bomber_paradise": {
        "name": "Bomber Paradise",
        "description": "Long, wide courses where distance dominates",
        "weights": {"sg_ott": 0.45, "sg_app": 0.30, "sg_arg": 0.10, "sg_putt": 0.15},
        "examples": ["Kapalua", "WM Phoenix Open"]
    },
    "classic_ballstriking": {
        "name": "Classic Ballstriking", 
        "description": "Premium on iron play and ball-striking",
        "weights": {"sg_ott": 0.20, "sg_app": 0.45, "sg_arg": 0.20, "sg_putt": 0.15},
        "examples": ["Riviera", "Memorial", "Genesis Invitational"]
    },
    "accuracy_premium": {
        "name": "Accuracy Premium",
        "description": "Tight, tree-lined with heavy rough",
        "weights": {"sg_ott": 0.20, "sg_app": 0.35, "sg_arg": 0.25, "sg_putt": 0.20},
        "examples": ["Harbour Town", "Colonial"]
    },
    "target_golf": {
        "name": "Target Golf",
        "description": "Strategic shot-making with hazards",
        "weights": {"sg_ott": 0.25, "sg_app": 0.35, "sg_arg": 0.25, "sg_putt": 0.15},
        "examples": ["TPC Sawgrass", "Bay Hill"]
    },
    "major_championship": {
        "name": "Major Championship",
        "description": "Severe setup with major pressure",
        "weights": {"sg_ott": 0.25, "sg_app": 0.35, "sg_arg": 0.25, "sg_putt": 0.15},
        "examples": ["Augusta", "US Open venues"]
    }
}

print("✓ Config loaded successfully")