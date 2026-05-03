"""Compatibility exports for older imports.

New workflow code should import deterministic rating from ``app.rating``.
"""

from app.rating import RatingTool

__all__ = ["RatingTool"]
