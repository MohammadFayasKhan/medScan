"""
test_ocr.py
===========
Unit tests for modules/ocr_engine.py — OCR text extraction and cleaning.

Note: Tests that require pytesseract or OpenCV are gracefully skipped
if those libraries are not installed.

Run with: pytest tests/test_ocr.py -v

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ocr_engine import (
    clean_ocr_output,
    extract_candidates,
    NON_MEDICINE_WORDS,
)


# ─────────────────────────────────────────────────────────────────────
# TESTS — CLEAN OCR OUTPUT
# ─────────────────────────────────────────────────────────────────────

def test_clean_ocr_output_removes_whitespace():
    """clean_ocr_output() should strip and collapse whitespace."""
    result = clean_ocr_output("  PARACETAMOL  \n\n  500mg  ")
    assert result == result.strip()
    assert "  " not in result  # no double spaces


def test_clean_ocr_output_handles_empty():
    """clean_ocr_output() should handle empty/None input safely."""
    assert clean_ocr_output("") == ""
    assert clean_ocr_output(None) == ""


def test_clean_ocr_output_fixes_common_errors():
    """clean_ocr_output() should fix digit→letter OCR errors in all-caps tokens."""
    result = clean_ocr_output("PARACETAM0L")  # 0 → O in caps token
    assert "PARACETAMOL" in result


def test_clean_ocr_output_preserves_normal_digits():
    """clean_ocr_output() should NOT change digits in mixed-case or numeric tokens."""
    result = clean_ocr_output("500mg dose")
    assert "500" in result  # digits in dosage should be preserved


def test_clean_ocr_output_removes_special_chars():
    """clean_ocr_output() should remove non-printable characters."""
    result = clean_ocr_output("TEST\x00\x01MEDICINE")
    assert "\x00" not in result
    assert "\x01" not in result


# ─────────────────────────────────────────────────────────────────────
# TESTS — CANDIDATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def test_extract_candidates_returns_list():
    """extract_candidates() should return a list."""
    result = extract_candidates("PARACETAMOL 500mg Store below 25C")
    assert isinstance(result, list)


def test_extract_candidates_filters_non_medicine_words():
    """extract_candidates() should exclude words in NON_MEDICINE_WORDS."""
    text = "TABLETS SYRUP CAPSULES Store below 25C"
    candidates = extract_candidates(text)
    for cand in candidates:
        assert cand.lower() not in NON_MEDICINE_WORDS


def test_extract_candidates_empty_input():
    """extract_candidates() should handle empty input."""
    result = extract_candidates("")
    assert result == []


def test_extract_candidates_returns_at_most_5():
    """extract_candidates() should return at most 5 candidates."""
    long_text = "ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA IOTA KAPPA"
    result = extract_candidates(long_text)
    assert len(result) <= 5


def test_extract_candidates_filters_short_tokens():
    """extract_candidates() should exclude tokens of 3 characters or fewer."""
    text = "THE FOR BY TO USE GEL PARACETAMOL"
    candidates = extract_candidates(text)
    for cand in candidates:
        assert len(cand) > 3


def test_extract_candidates_finds_medicine_name():
    """extract_candidates() should identify PARACETAMOL in typical OCR output."""
    text = "PARACETAMOL 500mg Tablets Store below 25 degrees"
    candidates = extract_candidates(text)
    # PARACETAMOL should be the best/first candidate
    assert len(candidates) > 0
    assert candidates[0].upper() == "PARACETAMOL"


# ─────────────────────────────────────────────────────────────────────
# TESTS — PIPELINE STRUCTURE (no Tesseract required)
# ─────────────────────────────────────────────────────────────────────

def test_run_ocr_pipeline_returns_dict_structure():
    """run_ocr_pipeline() with invalid input should return correct dict keys."""
    from modules.ocr_engine import run_ocr_pipeline
    import io

    # Create a fake "file" with invalid content
    fake_file = io.BytesIO(b"not_an_image")
    fake_file.name = "test.jpg"

    result = run_ocr_pipeline(fake_file)

    # Should return a dict with all required keys
    required_keys = {
        "success", "raw_text", "cleaned_text",
        "candidates", "best_candidate",
        "preprocessed_image", "original_image", "error"
    }
    assert required_keys.issubset(result.keys())

    # Should not crash — either success or graceful error
    assert isinstance(result["success"], bool)
    assert isinstance(result["candidates"], list)
