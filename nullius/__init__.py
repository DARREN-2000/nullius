"""Nullius - clinical decision support vertical slice.

Public surface kept intentionally small so the pipeline, API and tests all go
through the same entry points.
"""

from .app import Nullius, build_app

__all__ = ["Nullius", "build_app"]
__version__ = "2.0.0"
