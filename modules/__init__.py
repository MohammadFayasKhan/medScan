"""
__init__.py
===========
Package initialisation for the `modules` package of MEDSCAN AI.
Exposes package version and summary of all submodules for import convenience.

Exports:
  __version__  — current package version string
  __author__   — build identity

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

# ─────────────────────────────────────────────────────────────────────
# PACKAGE METADATA
# ─────────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__author__ = "ANTIGRAVITY BUILD"
__description__ = "MEDSCAN AI — Offline Medicine Intelligence System — Core Modules"

# Sub-module registry (informational, not imported to avoid circular deps)
__all__ = [
    "ocr_engine",
    "medicine_db",
    "medicine_search",
    "chatbot",
    "intent_classifier",
    "preprocessor",
    "model_trainer",
    "compare_engine",
    "export_utils",
]
