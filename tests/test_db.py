"""
test_db.py
==========
Unit tests for modules/medicine_db.py

These tests verify that the database loading and querying system works
correctly under normal conditions and handles edge cases gracefully.

Test coverage:
  - load_database() returns a properly structured DataFrame
  - All 27 required columns are present
  - List columns (brand_names, indications, etc.) are converted from CSV strings
  - validate_database() correctly identifies valid and invalid DataFrames
  - get_medicine_by_key() finds medicines by exact and case-insensitive name
  - get_all_names(), get_all_categories(), filter_by_category() work correctly
  - build_search_corpus() produces one document per medicine row

Run with: python -m pytest tests/test_db.py -v

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.medicine_db import (
    load_database,
    validate_database,
    get_medicine_by_key,
    get_all_names,
    get_all_categories,
    filter_by_category,
    build_search_corpus,
    REQUIRED_COLUMNS,
    LIST_COLUMNS,
)


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db():
    """Load database once for all tests in this module."""
    return load_database()


# ─────────────────────────────────────────────────────────────────────
# TESTS — LOADING
# ─────────────────────────────────────────────────────────────────────

def test_load_database_returns_dataframe(db):
    """load_database() should return a pandas DataFrame."""
    assert isinstance(db, pd.DataFrame)
    assert len(db) >= 20, "Database should have at least 20 medicines"


def test_database_has_required_columns(db):
    """DataFrame should contain all required columns."""
    for col in REQUIRED_COLUMNS:
        assert col in db.columns, f"Missing required column: {col}"


def test_list_columns_converted(db):
    """LIST_COLUMNS should be converted from strings to Python lists."""
    for col in LIST_COLUMNS:
        if col in db.columns:
            # At least some rows should have list values (not strings)
            sample = db[col].iloc[0]
            assert isinstance(sample, list), (
                f"Column '{col}' should be a list, got {type(sample)}"
            )


def test_name_lower_column_exists(db):
    """'name_lower' lookup column should be created by load_database()."""
    assert "name_lower" in db.columns


# ─────────────────────────────────────────────────────────────────────
# TESTS — VALIDATION
# ─────────────────────────────────────────────────────────────────────

def test_validate_database_valid(db):
    """validate_database() should return (True, []) for loaded database."""
    valid, missing = validate_database(db)
    assert valid is True
    assert missing == []


def test_validate_database_missing_column():
    """validate_database() should return (False, [...]) when columns are missing."""
    # Create a DataFrame with only some columns
    partial_df = pd.DataFrame({"name": ["test"], "generic_name": ["test_generic"]})
    valid, missing = validate_database(partial_df)
    assert valid is False
    assert len(missing) > 0


def test_validate_database_empty():
    """validate_database() on empty DataFrame should return invalid."""
    empty_df = pd.DataFrame()
    valid, missing = validate_database(empty_df)
    assert valid is False


# ─────────────────────────────────────────────────────────────────────
# TESTS — LOOKUP
# ─────────────────────────────────────────────────────────────────────

def test_get_medicine_by_key_found(db):
    """get_medicine_by_key() should find paracetamol by lowercase key."""
    med = get_medicine_by_key(db, "paracetamol")
    assert med is not None
    assert med["name"].lower() == "paracetamol"


def test_get_medicine_by_key_not_found(db):
    """get_medicine_by_key() should return None for unknown key."""
    med = get_medicine_by_key(db, "nonexistent_medicine_xyz")
    assert med is None


def test_get_medicine_by_key_case_insensitive(db):
    """get_medicine_by_key() should be case-insensitive."""
    med = get_medicine_by_key(db, "PARACETAMOL")
    assert med is not None


# ─────────────────────────────────────────────────────────────────────
# TESTS — COLLECTIONS
# ─────────────────────────────────────────────────────────────────────

def test_get_all_names_returns_list(db):
    """get_all_names() should return a non-empty sorted list."""
    names = get_all_names(db)
    assert isinstance(names, list)
    assert len(names) >= 20
    assert names == sorted(names)


def test_get_all_categories_returns_list(db):
    """get_all_categories() should return a non-empty list of unique categories."""
    cats = get_all_categories(db)
    assert isinstance(cats, list)
    assert len(cats) > 0
    # All categories should be unique
    assert len(cats) == len(set(cats))


def test_filter_by_category_nsaid(db):
    """filter_by_category() should filter to NSAID medicines."""
    filtered = filter_by_category(db, "nsaid")
    assert len(filtered) > 0
    # All filtered medicines should have 'nsaid' in their category
    for _, row in filtered.iterrows():
        assert "nsaid" in row["category"].lower()


def test_filter_by_category_all(db):
    """filter_by_category() with 'All' should return full DataFrame."""
    filtered = filter_by_category(db, "All")
    assert len(filtered) == len(db)


# ─────────────────────────────────────────────────────────────────────
# TESTS — CORPUS
# ─────────────────────────────────────────────────────────────────────

def test_build_search_corpus_correct_length(db):
    """build_search_corpus() should return one doc per medicine."""
    corpus = build_search_corpus(db)
    assert len(corpus) == len(db)


def test_build_search_corpus_non_empty_docs(db):
    """All documents in corpus should be non-empty strings."""
    corpus = build_search_corpus(db)
    for doc in corpus:
        assert isinstance(doc, str)
        assert len(doc) > 0
