"""
test_search.py
==============
Unit tests for modules/medicine_search.py — multi-strategy search engine.

Run with: pytest tests/test_search.py -v

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.medicine_db import load_database, build_search_corpus
from modules.medicine_search import (
    build_search_index,
    exact_search,
    tfidf_search,
    fuzzy_search,
    get_suggestions,
    search_medicine,
)


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db():
    """Load the medicine database once."""
    return load_database()


@pytest.fixture(scope="module")
def search_index(db):
    """Build search index from loaded database."""
    corpus = build_search_corpus(db)
    return build_search_index(corpus)


# ─────────────────────────────────────────────────────────────────────
# TESTS — EXACT SEARCH
# ─────────────────────────────────────────────────────────────────────

def test_exact_search_finds_paracetamol(db):
    """exact_search() should find paracetamol by exact name."""
    result = exact_search(db, "paracetamol")
    assert result is not None
    assert result["name"].lower() == "paracetamol"


def test_exact_search_case_insensitive(db):
    """exact_search() should be case-insensitive."""
    result = exact_search(db, "PARACETAMOL")
    assert result is not None
    assert "paracetamol" in result["name"].lower()


def test_brand_name_search_eyemist(db):
    """Searching 'Eyemist' brand name should resolve to hypromellose via synonym."""
    result = exact_search(db, "eyemist")
    assert result is not None
    assert "hypromellose" in result["name"].lower()


# ─────────────────────────────────────────────────────────────────────
# TESTS — TFIDF SEARCH
# ─────────────────────────────────────────────────────────────────────

def test_tfidf_search_returns_list(db, search_index):
    """tfidf_search() should return a list."""
    vec, mat = search_index
    results = tfidf_search("fever headache pain", vec, mat, db)
    assert isinstance(results, list)


def test_tfidf_search_finds_relevant(db, search_index):
    """tfidf_search() for 'fever pain' should find paracetamol or ibuprofen."""
    vec, mat = search_index
    results = tfidf_search("fever headache pain tablet", vec, mat, db, top_n=3)
    if results:
        names = [r["medicine"]["name"].lower() for r in results]
        found_relevant = any(
            n in names for n in ["paracetamol", "ibuprofen", "aspirin", "aceclofenac"]
        )
        assert found_relevant, f"Expected pain/fever medicine in results, got: {names}"


# ─────────────────────────────────────────────────────────────────────
# TESTS — FUZZY SEARCH
# ─────────────────────────────────────────────────────────────────────

def test_fuzzy_search_handles_typo(db):
    """fuzzy_search() should find paracetamol when query has a typo."""
    results = fuzzy_search(db, "paracetaml")
    assert len(results) > 0
    assert "paracetamol" in results[0]["medicine"]["name"].lower()


def test_fuzzy_search_handles_ocr_error(db):
    """fuzzy_search() should handle OCR-style 0→O substitution."""
    results = fuzzy_search(db, "paracetam0l")
    # Should still find paracetamol
    assert isinstance(results, list)
    if results:
        assert "paracetamol" in results[0]["medicine"]["name"].lower()


def test_fuzzy_search_returns_list(db):
    """fuzzy_search() should always return a list, even for poor queries."""
    results = fuzzy_search(db, "completely_made_up_xxzz")
    assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────
# TESTS — MASTER SEARCH
# ─────────────────────────────────────────────────────────────────────

def test_search_medicine_returns_dict(db, search_index):
    """search_medicine() should return a dict with required keys."""
    vec, mat = search_index
    result = search_medicine("paracetamol", db, vec, mat)
    assert isinstance(result, dict)
    required_keys = {"found", "medicine", "strategy", "confidence", "suggestions", "search_time_ms"}
    assert required_keys.issubset(result.keys())


def test_search_medicine_found_with_exact(db, search_index):
    """search_medicine() should find paracetamol with exact strategy."""
    vec, mat = search_index
    result = search_medicine("paracetamol", db, vec, mat)
    assert result["found"] is True
    assert result["strategy"] in ("exact", "tfidf", "fuzzy")
    assert result["medicine"]["name"].lower() == "paracetamol"


def test_search_medicine_not_found_returns_suggestions(db, search_index):
    """search_medicine() for a nonsense query should return suggestions list."""
    vec, mat = search_index
    result = search_medicine("zzzzxxxyyy", db, vec, mat)
    assert result["found"] is False
    assert isinstance(result["suggestions"], list)


def test_search_medicine_empty_query(db, search_index):
    """search_medicine() with empty query should return not-found safely."""
    vec, mat = search_index
    result = search_medicine("", db, vec, mat)
    assert result["found"] is False


def test_get_suggestions_returns_list(db, search_index):
    """get_suggestions() should return a list of medicine name strings."""
    vec, mat = search_index
    suggestions = get_suggestions(db, "paracetaml", vec, mat)
    assert isinstance(suggestions, list)
    assert len(suggestions) <= 5
