"""
medicine_search.py
==================
This module implements the three-strategy medicine search engine.

Why three strategies?
  A single search approach would fail in many real-world cases:
  - "paracetamol" → exact match works perfectly
  - "fever headache tablet" → needs TF-IDF semantic similarity
  - "paracetaml" (OCR typo) → needs Levenshtein fuzzy matching

Strategy order (falls through to next if confidence is low):
  1. EXACT MATCH   — checks name_lower and brand name synonyms (O(1) lookup)
  2. TF-IDF MATCH  — vectorises the query and finds the closest medicine
                     in the TF-IDF document space using cosine similarity
  3. FUZZY MATCH   — uses Levenshtein edit distance to find the closest
                     medicine name, tolerating up to ~30% character changes

This cascade ensures high recall (few missed medicines) while the confidence
score tells the UI how certain we are about the match.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import time
import json
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# scikit-learn: TF-IDF vectoriser and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# fuzzywuzzy: Levenshtein-based fuzzy string matching
from fuzzywuzzy import fuzz, process as fuzz_process

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Minimum cosine similarity score (0–1) to accept a TF-IDF match
TFIDF_THRESHOLD = 0.15

# Minimum fuzzywuzzy score (0–100) to accept a fuzzy match
FUZZY_THRESHOLD = 62

# Maximum number of "did you mean?" suggestions to return
MAX_SUGGESTIONS = 5

# TF-IDF n-gram range: unigrams + bigrams improve multi-word searches
NGRAM_RANGE = (1, 2)

# Path to synonyms file (brand → generic mapping)
SYNONYMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "synonyms.json"
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# SYNONYM LOADING
# ─────────────────────────────────────────────────────────────────────

def load_synonyms(path: str = SYNONYMS_PATH) -> dict:
    """
    Load brand-name → generic-name synonym mapping from JSON file.

    Used to resolve brand names (e.g., "Eyemist") to their database
    primary key (e.g., "hypromellose") before searching.

    Args:
        path (str): Path to synonyms.json. Defaults to SYNONYMS_PATH.

    Returns:
        dict: Mapping of lowercase brand name → lowercase generic name.
              Returns empty dict if file not found.

    Example:
        >>> syns = load_synonyms()
        >>> syns["eyemist"]
        'hypromellose'
    """
    if not os.path.exists(path):
        logger.warning(f"synonyms.json not found at {path} — brand name lookup disabled")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Normalise all keys and values to lowercase for consistent lookup
        return {k.lower().strip(): v.lower().strip() for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Could not load synonyms.json: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────
# SEARCH INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_search_index(corpus: list) -> tuple:
    """
    Build a TF-IDF search index from the medicine text corpus.

    Called once at app startup and results cached in session state.
    The index enables fast semantic similarity search across all medicines.

    Algorithm:
      - TF-IDF (Term Frequency–Inverse Document Frequency) represents
        each medicine as a weighted vector where rare important terms
        get higher weights than common words.
      - Bigrams (word pairs) improve multi-word search queries.

    Args:
        corpus (list[str]): List of text documents, one per medicine.
                            Built by medicine_db.build_search_corpus().

    Returns:
        tuple[TfidfVectorizer, scipy.sparse.matrix]:
            (fitted_vectorizer, tfidf_matrix)
            Vectorizer transforms new queries; matrix represents all medicines.

    Example:
        >>> vectorizer, matrix = build_search_index(corpus)
        >>> matrix.shape[0] == len(corpus)
        True
    """
    # Configure TF-IDF vectoriser with bigrams and Unicode normalisation
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,      # unigrams + bigrams for richer matching
        analyzer="word",               # word-level analysis (not char-level)
        strip_accents="unicode",       # normalise accented characters
        min_df=1,                      # include terms appearing in ≥1 document
        max_features=10000             # cap vocabulary size for memory efficiency
    )

    # Fit vectoriser on corpus and transform to TF-IDF matrix
    # Matrix shape: (num_medicines, vocabulary_size)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    logger.info(f"Search index built: {tfidf_matrix.shape[0]} docs, {tfidf_matrix.shape[1]} features")
    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────────────────────────────
# STRATEGY 1: EXACT STRING MATCHING
# ─────────────────────────────────────────────────────────────────────

def exact_search(df: pd.DataFrame, query: str, synonyms: dict = None) -> dict:
    if not query:
        return None

    if synonyms is None:
        synonyms = load_synonyms()

    q = query.lower().strip()

    # Synonym resolution
    if q in synonyms:
        resolved = synonyms[q]
        result = exact_search(df, resolved, synonyms={})
        if result:
            logger.debug(f"Synonym resolved: '{q}' → '{resolved}'")
            return result

    # Exact name match
    exact = df[df["name_lower"] == q]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # Prefix match
    prefix = df[df["name_lower"].str.startswith(q, na=False)]
    if not prefix.empty:
        return prefix.iloc[0].to_dict()

    # Substring
    contains = df[df["name_lower"].str.contains(q, na=False, regex=False)]
    if not contains.empty:
        return contains.iloc[0].to_dict()

    # Generic name
    generic = df[df["generic_name"].str.lower().str.contains(q, na=False, regex=False)]
    if not generic.empty:
        return generic.iloc[0].to_dict()

    return None


# ─────────────────────────────────────────────────────────────────────
# STRATEGY 2: TF-IDF COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────

def tfidf_search(query: str, vectorizer, matrix,
                 df: pd.DataFrame, top_n: int = 3) -> list:
    if vectorizer is None or matrix is None:
        return []

    try:
        query_vector = vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < TFIDF_THRESHOLD:
                continue

            medicine = df.iloc[int(idx)].to_dict()
            results.append({"medicine": medicine, "score": score})

        return results
    except Exception as e:
        logger.warning(f"TF-IDF search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# STRATEGY 3: FUZZY STRING MATCHING
# ─────────────────────────────────────────────────────────────────────

def fuzzy_search(df: pd.DataFrame, query: str) -> list:
    """
    Fuzzy string matching on the 11k database.
    Since 11,800 rows is expensive for full Levenshtein, this is kept simple.
    """
    if not query or len(query) < 4:
        return []  # Fuzzy search on <4 chars across 11k DB is useless/noisy

    # Use rapidfuzz/fuzzywuzzy to extract best from name col
    name_strings = df['name'].tolist()
    
    matches = fuzz_process.extractBests(
        query,
        name_strings,
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_THRESHOLD,
        limit=5
    )

    if not matches:
        return []

    results = []
    seen = set()
    for matched_name, score in matches:
        # Get first df item that matches this name
        row = df[df['name'] == matched_name]
        if not row.empty:
            idx = row.index[0]
            if idx not in seen:
                results.append({
                    "medicine": row.iloc[0].to_dict(),
                    "score": score,
                    "matched_on": "name"
                })
                seen.add(idx)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────
# SUGGESTION ENGINE
# ─────────────────────────────────────────────────────────────────────

def get_suggestions(df: pd.DataFrame, query: str,
                    vectorizer=None, matrix=None) -> list:
    """
    Get autocomplete / "Did you mean?" suggestions for failed searches.

    Combines fuzzy search results and TF-IDF results to generate diverse
    suggestions when the primary search fails.

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.
        query (str): The failed search query.
        vectorizer: Fitted TF-IDF vectoriser (or None).
        matrix: TF-IDF matrix (or None).

    Returns:
        list[str]: Up to MAX_SUGGESTIONS medicine name strings.

    Example:
        >>> sug = get_suggestions(df, "paracetaml", vectorizer, matrix)
        >>> "paracetamol" in sug
        True
    """
    suggestion_names = []

    # Source 1: Fuzzy search (top 3 — best for typos/OCR errors)
    fuzzy_results = fuzzy_search(df, query)
    for r in fuzzy_results[:3]:
        name = r["medicine"].get("name", "")
        if name and name not in suggestion_names:
            suggestion_names.append(name)

    # Source 2: TF-IDF search (top 2 — best for semantic/description queries)
    if vectorizer is not None and matrix is not None:
        tfidf_results = tfidf_search(query, vectorizer, matrix, df, top_n=2)
        for r in tfidf_results:
            name = r["medicine"].get("name", "")
            if name and name not in suggestion_names:
                suggestion_names.append(name)

    # Return deduplicated list capped at MAX_SUGGESTIONS
    return suggestion_names[:MAX_SUGGESTIONS]


# ─────────────────────────────────────────────────────────────────────
# MASTER SEARCH FUNCTION
# ─────────────────────────────────────────────────────────────────────

def search_medicine(query: str, df: pd.DataFrame,
                    vectorizer=None, matrix=None) -> dict:
    """
    Master search function: applies all 3 strategies in priority order.

    Strategy cascade:
      1. Exact match + synonym resolution (confidence 1.0 on hit)
      2. TF-IDF cosine similarity (confidence = cosine score)
      3. Fuzzy Levenshtein matching (confidence = fuzz score / 100)
      If none succeed → returns not-found with suggestions.

    Args:
        query (str): Medicine name or description to search for.
        df (pd.DataFrame): Processed medicine DataFrame.
        vectorizer: Fitted TF-IDF vectoriser (from build_search_index or None).
        matrix: TF-IDF matrix (from build_search_index or None).

    Returns:
        dict: Comprehensive search result containing:
            {
              "found": bool,
              "medicine": dict | None,
              "strategy": str,       # "exact" | "tfidf" | "fuzzy" | "none"
              "confidence": float,   # 0.0–1.0
              "suggestions": list,   # medicine names if not found
              "search_time_ms": float
            }

    Example:
        >>> result = search_medicine("paracetamol", df, vec, mat)
        >>> result["found"]
        True
        >>> result["strategy"]
        'exact'
    """
    # Record start time for performance measurement
    start_time = time.perf_counter()

    # Default "not found" result template
    not_found_result = {
        "found": False,
        "medicine": None,
        "strategy": "none",
        "confidence": 0.0,
        "suggestions": [],
        "search_time_ms": 0.0
    }

    # Input validation
    if not query or not isinstance(query, str):
        not_found_result["search_time_ms"] = (time.perf_counter() - start_time) * 1000
        return not_found_result

    # Clean query: strip, but preserve case for fuzzy matching
    query = query.strip()
    # Load synonyms once for this search operation
    synonyms = load_synonyms()

    # ── Strategy 1: Exact / Synonym search ──────────────────────────
    exact_result = exact_search(df, query, synonyms=synonyms)
    if exact_result:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Search '{query}' → exact hit in {elapsed_ms:.1f}ms")
        return {
            "found": True,
            "medicine": exact_result,
            "strategy": "exact",
            "confidence": 1.0,
            "suggestions": [],
            "search_time_ms": elapsed_ms
        }

    # ── Strategy 2: TF-IDF Cosine Similarity ────────────────────────
    if vectorizer is not None and matrix is not None:
        tfidf_results = tfidf_search(query, vectorizer, matrix, df, top_n=1)
        if tfidf_results:
            best = tfidf_results[0]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Search '{query}' → TF-IDF hit (score={best['score']:.3f}) in {elapsed_ms:.1f}ms")
            return {
                "found": True,
                "medicine": best["medicine"],
                "strategy": "tfidf",
                "confidence": float(best["score"]),
                "suggestions": [],
                "search_time_ms": elapsed_ms
            }

    # ── Strategy 3: Fuzzy String Matching ───────────────────────────
    fuzzy_results = fuzzy_search(df, query)
    if fuzzy_results:
        best = fuzzy_results[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # Normalise fuzz score (0–100) to confidence (0.0–1.0)
        confidence = best["score"] / 100.0
        logger.info(f"Search '{query}' → fuzzy hit '{best['medicine']['name']}' (score={best['score']}) in {elapsed_ms:.1f}ms")
        return {
            "found": True,
            "medicine": best["medicine"],
            "strategy": "fuzzy",
            "confidence": confidence,
            "suggestions": [],
            "search_time_ms": elapsed_ms
        }

    # ── All strategies failed → return suggestions ───────────────────
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    suggestions = get_suggestions(df, query, vectorizer, matrix)
    logger.info(f"Search '{query}' → no result in {elapsed_ms:.1f}ms. Suggestions: {suggestions}")

    return {
        "found": False,
        "medicine": None,
        "strategy": "none",
        "confidence": 0.0,
        "suggestions": suggestions,
        "search_time_ms": elapsed_ms
    }
