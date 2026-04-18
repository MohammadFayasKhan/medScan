"""
modules/__init__.py
===================
Package marker for the modules/ directory.

The modules/ package contains all the core AI and NLP logic for MedScan AI.
Each sub-module is imported independently by app.py and components/ as needed.

Package contents:
  preprocessor       - NLP text cleaning and tokenisation
  medicine_db        - CSV database loading and querying
  medicine_search    - 3-strategy medicine search engine
  ocr_engine         - OpenCV + Tesseract OCR pipeline
  chatbot            - TF-IDF intent classification and response generation
  intent_classifier  - Naive Bayes classifier training and inference
  compare_engine     - Medicine comparison scoring and radar chart
  model_trainer      - Model persistence (load/save via joblib)
  export_utils       - CSV, JSON, and text export helpers

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


# ─────────────────────────────────────────────────────────────────────
# PACKAGE METADATA
# ─────────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__author__ = "MedScan AI Project"
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
