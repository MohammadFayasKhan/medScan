"""
components/__init__.py
======================
Package marker for the components/ directory.

The components/ package contains all Streamlit UI rendering code.
Each module is responsible for one section of the interface.
Logic (AI, search, NLP) is intentionally kept separate in modules/.

This separation makes it easy to:
  - Change the UI without touching the AI logic
  - Test the AI logic independently of Streamlit
  - Read and understand each part of the app in isolation

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


__version__ = "1.0.0"
__all__ = [
    "ui_styles",
    "medicine_card",
    "info_sections",
    "chatbot_ui",
    "compare_ui",
    "sidebar_ui",
    "scan_ui",
]
