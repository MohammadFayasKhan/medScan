"""
test_algorithms.py
==================
ML model accuracy and algorithm correctness tests.

Tests:
  - Naive Bayes classifier accuracy (should be >85%)
  - TF-IDF vectoriser properties
  - Compare engine scoring
  - Preprocessor pipeline correctness

Run with: pytest tests/test_algorithms.py -v

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────
# PREPROCESSOR TESTS
# ─────────────────────────────────────────────────────────────────────

def test_preprocessor_clean_text():
    """clean_text() should lowercase and remove special characters."""
    from modules.preprocessor import clean_text
    result = clean_text("What ARE the Side-Effects?!")
    assert result == "what are the side effects"


def test_preprocessor_pipeline_output():
    """preprocess_pipeline() should return a non-empty string for valid input."""
    from modules.preprocessor import preprocess_pipeline
    result = preprocess_pipeline("What are the side effects of this medicine?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_medical_keywords():
    """extract_medical_keywords() should find known medical terms."""
    from modules.preprocessor import extract_medical_keywords
    keywords = extract_medical_keywords("Is this safe during pregnancy for the child?")
    assert "pregnancy" in keywords or "safe" in keywords


# ─────────────────────────────────────────────────────────────────────
# NAIVE BAYES ACCURACY TEST
# ─────────────────────────────────────────────────────────────────────

def test_naive_bayes_training_accuracy():
    """Naive Bayes classifier should achieve ≥85% training accuracy."""
    from modules.chatbot import load_intents
    from modules.intent_classifier import (
        prepare_training_data, train_classifier
    )
    from sklearn.metrics import accuracy_score

    intents = load_intents()
    X, y = prepare_training_data(intents)
    vectorizer, classifier, label_encoder = train_classifier(X, y)

    X_vec = vectorizer.transform(X)
    y_enc = label_encoder.transform(y)
    preds = classifier.predict(X_vec)

    accuracy = accuracy_score(y_enc, preds)
    assert accuracy >= 0.85, f"Expected ≥85% accuracy, got {accuracy:.1%}"


def test_naive_bayes_predict_known_intent():
    """Naive Bayes should correctly predict a clear intent from training data."""
    from modules.chatbot import load_intents
    from modules.intent_classifier import (
        prepare_training_data, train_classifier, predict_intent
    )

    intents = load_intents()
    X, y = prepare_training_data(intents)
    vectorizer, classifier, label_encoder = train_classifier(X, y)

    label, prob = predict_intent(
        "what are the side effects",
        vectorizer, classifier, label_encoder
    )
    assert label == "side_effects"
    assert prob > 0.5


# ─────────────────────────────────────────────────────────────────────
# TFIDF PROPERTIES TESTS
# ─────────────────────────────────────────────────────────────────────

def test_tfidf_vectorizer_transforms_query():
    """TF-IDF vectoriser should transform a query without error."""
    from modules.medicine_db import load_database, build_search_corpus
    from modules.medicine_search import build_search_index

    df = load_database()
    corpus = build_search_corpus(df)
    vec, mat = build_search_index(corpus)

    query_vec = vec.transform(["paracetamol fever"])
    assert query_vec.shape == (1, mat.shape[1])


def test_cosine_similarity_range():
    """Cosine similarity scores should be in [0, 1]."""
    from sklearn.metrics.pairwise import cosine_similarity
    from modules.medicine_db import load_database, build_search_corpus
    from modules.medicine_search import build_search_index

    df = load_database()
    corpus = build_search_corpus(df)
    vec, mat = build_search_index(corpus)

    q_vec = vec.transform(["paracetamol"])
    sims = cosine_similarity(q_vec, mat).flatten()

    assert np.all(sims >= 0.0)
    assert np.all(sims <= 1.001)  # small float tolerance


# ─────────────────────────────────────────────────────────────────────
# COMPARE ENGINE TESTS
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def two_medicines():
    from modules.medicine_db import load_database, get_medicine_by_key
    df = load_database()
    para = get_medicine_by_key(df, "paracetamol")
    ibu = get_medicine_by_key(df, "ibuprofen")
    return [para, ibu]


def test_compute_medicine_scores_structure(two_medicines):
    """compute_medicine_scores() should return a dict with all dimensions."""
    from modules.compare_engine import compute_medicine_scores, COMPARISON_DIMENSIONS
    scores = compute_medicine_scores(two_medicines[0])
    for dim in COMPARISON_DIMENSIONS:
        assert dim in scores
        assert isinstance(scores[dim], (int, float))


def test_compare_medicines_returns_dict(two_medicines):
    """compare_medicines() should return a dict with scores and winners."""
    from modules.compare_engine import compare_medicines
    result = compare_medicines(two_medicines)
    assert "scores" in result
    assert "winners" in result
    assert "medicines" in result
    assert len(result["scores"]) == 2


def test_compare_medicines_requires_two():
    """compare_medicines() should raise ValueError with only 1 medicine."""
    from modules.compare_engine import compare_medicines
    with pytest.raises(ValueError):
        compare_medicines([{"name": "test"}])


def test_radar_chart_returns_figure(two_medicines):
    """generate_radar_chart() should return a matplotlib Figure."""
    import matplotlib.pyplot as plt
    from modules.compare_engine import (
        generate_radar_chart, compute_medicine_scores
    )
    scores = [compute_medicine_scores(m) for m in two_medicines]
    fig = generate_radar_chart(two_medicines, scores)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
