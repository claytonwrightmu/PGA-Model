from __future__ import annotations

from typing import Dict

from config import COURSE_ARCHETYPES


def get_archetypes() -> Dict:
    """
    Single source of truth is config.COURSE_ARCHETYPES.
    This wrapper makes course_engine explicit and testable.
    """
    return COURSE_ARCHETYPES
