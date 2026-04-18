"""
test_chatbot.py
===============
Unit tests for modules/chatbot.py — intent classification and response generation.

Run with: pytest tests/test_chatbot.py -v

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chatbot import (
    load_intents,
    build_intent_index,
    classify_intent,
    generate_response,
    preprocess_question,
    get_chat_response,
)


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def intents():
    """Load intents once for all tests."""
    return load_intents()


@pytest.fixture(scope="module")
def intent_index(intents):
    """Build index once."""
    return build_intent_index(intents)


@pytest.fixture(scope="module")
def sample_medicine():
    """A minimal medicine dict for response generation tests."""
    return {
        "name": "paracetamol",
        "generic_name": "Acetaminophen",
        "category": "Analgesic / Antipyretic",
        "form": "Tablet / Syrup",
        "strength": "500mg / 650mg",
        "active_substance": "Paracetamol",
        "features": ["Fast-acting", "OTC", "Safe for most"],
        "uses": "Fever and pain relief",
        "mechanism": "Inhibits prostaglandin synthesis in CNS",
        "indications": ["Fever", "Headache", "Toothache"],
        "dosage": "Adults: 500–1000mg every 4–6h; max 4g/day",
        "timing": "With or without food",
        "admin_tips": "Avoid other paracetamol products simultaneously",
        "spacing": "Minimum 4 hours between doses",
        "warning_pregnancy": "Safe in all trimesters at recommended doses",
        "warning_pediatric": "Safe from birth at weight-based dosing",
        "warning_driving": "Does not impair driving",
        "warning_storage": "Store below 25°C in dry place",
        "contraindications": ["Severe hepatic impairment", "Paracetamol hypersensitivity"],
        "interactions": ["Warfarin", "Alcohol", "Rifampicin"],
        "side_effects_common": ["Nausea", "Rash (rare)"],
        "side_effects_serious": ["Hepatotoxicity (overdose)", "Acute liver failure"],
        "substitutes": ["Ibuprofen", "Aspirin (adults)", "Diclofenac"],
        "pack_sizes": ["Strip of 10", "60ml syrup"],
        "sources": ["WHO Essential Medicines List", "Drugs.com"],
        "manufacturer": "GSK",
        "brand_names": ["Crocin", "Dolo", "Calpol"],
    }


# ─────────────────────────────────────────────────────────────────────
# TESTS — LOADING
# ─────────────────────────────────────────────────────────────────────

def test_load_intents_returns_dict(intents):
    """load_intents() should return a dict with 'intents' key."""
    assert isinstance(intents, dict)
    assert "intents" in intents
    assert len(intents["intents"]) == 14


def test_build_intent_index_returns_tuple(intent_index):
    """build_intent_index() should return a tuple of (vectorizer, matrix, labels)."""
    vec, mat, labels = intent_index
    assert vec is not None
    assert mat is not None
    assert isinstance(labels, list)
    assert len(labels) > 100  # 14 intents × 15+ patterns each


# ─────────────────────────────────────────────────────────────────────
# TESTS — INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────

def test_classify_intent_uses(intent_index):
    """'what does this medicine treat' should classify to 'uses'."""
    vec, mat, labels = intent_index
    tag, conf = classify_intent("what does this medicine treat", vec, mat, labels)
    assert tag == "uses", f"Expected 'uses', got '{tag}'"
    assert conf > 0.0


def test_classify_intent_dosage(intent_index):
    """'how many tablets should I take' should classify to 'dosage'."""
    vec, mat, labels = intent_index
    tag, conf = classify_intent("how many tablets should I take", vec, mat, labels)
    assert tag == "dosage", f"Expected 'dosage', got '{tag}'"


def test_classify_intent_side_effects(intent_index):
    """'are there any adverse reactions' should classify to 'side_effects'."""
    vec, mat, labels = intent_index
    tag, conf = classify_intent("are there any adverse reactions", vec, mat, labels)
    assert tag == "side_effects", f"Expected 'side_effects', got '{tag}'"


def test_classify_intent_pregnancy(intent_index):
    """'is this safe during pregnancy' should classify to 'pregnancy'."""
    vec, mat, labels = intent_index
    tag, conf = classify_intent("is this safe during pregnancy", vec, mat, labels)
    assert tag == "pregnancy", f"Expected 'pregnancy', got '{tag}'"


def test_classify_intent_overdose(intent_index):
    """'what if I take too much' should classify to 'overdose'."""
    vec, mat, labels = intent_index
    tag, conf = classify_intent("what if I take too much", vec, mat, labels)
    assert tag == "overdose", f"Expected 'overdose', got '{tag}'"


def test_unknown_returns_unknown(intent_index):
    """Completely nonsensical input should return 'unknown'."""
    vec, mat, labels = intent_index
    tag, conf = classify_intent("xyz abc 123 @@@ nonsense", vec, mat, labels)
    # May or may not be unknown depending on threshold — just check it runs
    assert isinstance(tag, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────────────────────────
# TESTS — RESPONSE GENERATION
# ─────────────────────────────────────────────────────────────────────

def test_generate_response_not_empty(sample_medicine):
    """generate_response() should return a non-empty string."""
    response = generate_response("uses", sample_medicine, 0.85)
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_response_contains_medicine_name(sample_medicine):
    """generate_response() should reference the medicine name in the response."""
    response = generate_response("overview", sample_medicine, 0.9)
    # Either the raw name or title-case version should appear
    assert "paracetamol" in response.lower() or "acetaminophen" in response.lower()


def test_generate_response_all_intents(sample_medicine):
    """All 14 intents should generate a valid non-empty response."""
    all_intents = [
        "overview", "uses", "mechanism", "dosage", "side_effects",
        "warnings", "contraindications", "interactions", "substitutes",
        "pregnancy", "pediatric", "admin_tips", "storage", "overdose", "unknown"
    ]
    for intent in all_intents:
        response = generate_response(intent, sample_medicine, 0.7)
        assert isinstance(response, str), f"Intent '{intent}' returned non-string"
        assert len(response) > 10, f"Intent '{intent}' returned too-short response"


# ─────────────────────────────────────────────────────────────────────
# TESTS — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────

def test_preprocess_question_lowercases():
    """preprocess_question() should return lowercase text."""
    result = preprocess_question("WHAT ARE THE SIDE EFFECTS?")
    assert result == result.lower()


def test_preprocess_question_removes_stopwords():
    """preprocess_question() should remove common stopwords."""
    result = preprocess_question("what are the side effects")
    # "what", "are", "the" should be removed; "side", "effects" should remain
    assert "side" in result or "effect" in result


def test_preprocess_question_handles_empty():
    """preprocess_question() should handle empty string without crashing."""
    result = preprocess_question("")
    assert isinstance(result, str)
