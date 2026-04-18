"""
medicine_search.py
==================
Multi-strategy offline medicine search engine.

Three search strategies applied in cascading priority order:
  Strategy 1: Exact string matching (fastest, O(n))
  Strategy 2: TF-IDF + Cosine Similarity (semantic matching)
  Strategy 3: Fuzzy string matching (handles typos and OCR errors)

The master search function (search_medicine) tries all three in order
and returns on the first strategy that produces a confident match.

Imports from: sklearn, fuzzywuzzy, numpy, pandas, time, modules/medicine_db
Exports:      build_search_index, search_medicine, get_suggestions

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
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
    """
    Try exact and prefix string matches on medicine names and brand names.

    This is the fastest strategy — O(n) linear scan through the DataFrame.
    Should be tried first before more expensive TF-IDF or fuzzy searches.

    Search priority:
      1. Synonym lookup (brand → generic resolution)
      2. Exact match on name_lower
      3. name_lower starts with query
      4. Query is a substring of name_lower
      5. Generic name contains query
      6. Any brand name contains query

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.
        query (str): User input string to search for.
        synonyms (dict): Brand→generic name mapping. If None, loads fresh.

    Returns:
        dict | None: Full medicine record dict if found, else None.

    Example:
        >>> result = exact_search(df, "paracetamol")
        >>> result["name"]
        'paracetamol'
    """
    if not query:
        return None

    # Load synonyms if not provided
    if synonyms is None:
        synonyms = load_synonyms()

    # Normalise: lowercase, strip, remove extra spaces
    q = query.lower().strip()

    # Priority 0: Synonym resolution — e.g., "eyemist" → "hypromellose"
    if q in synonyms:
        resolved = synonyms[q]
        result = exact_search(df, resolved, synonyms={})  # recursive with resolved key
        if result:
            logger.debug(f"Synonym resolved: '{q}' → '{resolved}'")
            return result

    # Priority 1: Exact name match
    exact = df[df["name_lower"] == q]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # Priority 2: Name starts with query (prefix match)
    prefix = df[df["name_lower"].str.startswith(q, na=False)]
    if not prefix.empty:
        return prefix.iloc[0].to_dict()

    # Priority 3: Name contains query as substring
    contains = df[df["name_lower"].str.contains(q, na=False, regex=False)]
    if not contains.empty:
        return contains.iloc[0].to_dict()

    # Priority 4: Generic name (lowercase) contains query
    generic = df[df["generic_name"].str.lower().str.contains(q, na=False, regex=False)]
    if not generic.empty:
        return generic.iloc[0].to_dict()

    # Priority 5: Any brand name in brand_names list contains query
    for _, row in df.iterrows():
        brands = row.get("brand_names", [])
        if isinstance(brands, list):
            if any(q in b.lower() for b in brands):
                return row.to_dict()
        elif q in str(brands).lower():
            return row.to_dict()

    return None


# ─────────────────────────────────────────────────────────────────────
# STRATEGY 2: TF-IDF COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────

def tfidf_search(query: str, vectorizer, matrix,
                 df: pd.DataFrame, top_n: int = 3) -> list:
    """
    Semantic search using TF-IDF cosine similarity.

    Transforms the query into its TF-IDF vector representation and computes
    cosine similarity against all medicine vectors in the index.
    Higher similarity score = more semantically related to query.

    Formula: similarity(A, B) = (A·B) / (|A| × |B|)
    where A = query vector, B = medicine document vector.

    Args:
        query (str): Search query string.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectoriser from build_search_index.
        matrix: TF-IDF matrix (scipy sparse) from build_search_index.
        df (pd.DataFrame): Processed medicine DataFrame (for retrieving records).
        top_n (int): Number of top results to return. Default 3.

    Returns:
        list[dict]: List of {medicine_dict, score} dicts sorted by score descending.
                    Only results with score > TFIDF_THRESHOLD are included.

    Example:
        >>> results = tfidf_search("fever headache", vectorizer, matrix, df)
        >>> results[0]["score"] > 0.15
        True
    """
    if vectorizer is None or matrix is None:
        return []

    try:
        # Transform query string into TF-IDF vector (1 × vocab_size matrix)
        query_vector = vectorizer.transform([query.lower()])

        # Compute cosine similarity between query and all medicine documents
        # Result shape: (1, num_medicines) → flatten to 1D array
        similarities = cosine_similarity(query_vector, matrix).flatten()

        # Get indices of top_n highest similarity scores (descending order)
        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])

            # Only include results above the similarity threshold
            if score < TFIDF_THRESHOLD:
                continue

            # Retrieve the medicine record at this index
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
    Fuzzy string matching to handle OCR errors, typos, and near-misses.

    Uses fuzzywuzzy's Levenshtein distance metric to find medicine names
    that are similar (but not identical) to the query string.
    Searches across all name variants: medicine names, brand names, generics.

    Example OCR errors handled:
      "paracetaml" → "paracetamol" (missing letter)
      "paracetam0l" → "paracetamol" (0 instead of O)
      "ibuprofin" → "ibuprofen" (spelling error)

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.
        query (str): Search string (possibly with typos/OCR errors).

    Returns:
        list[dict]: List of {medicine, score, matched_on} dicts,
                    sorted by score descending.
                    Only results with score > FUZZY_THRESHOLD included.

    Example:
        >>> results = fuzzy_search(df, "paracetaml")
        >>> results[0]["medicine"]["name"]
        'paracetamol'
    """
    if not query:
        return []

    # Build a flat list of all name variants with their back-references
    # Each entry: (display_name, medicine_df_index, matched_field)
    name_list = []
    for idx, row in df.iterrows():
        # Add primary name
        name_list.append((str(row["name"]), idx, "name"))

        # Add generic name
        if row.get("generic_name"):
            name_list.append((str(row["generic_name"]).split("/")[0].strip(), idx, "generic_name"))

        # Add each brand name individually
        brands = row.get("brand_names", [])
        if isinstance(brands, list):
            for brand in brands[:3]:  # limit to first 3 brands per medicine
                if brand and len(brand) > 3:
                    name_list.append((brand.strip(), idx, "brand_names"))

    # Extract just the name strings for fuzzywuzzy
    name_strings = [n[0] for n in name_list]

    # Run fuzzy extraction: finds best matches above score cutoff
    # token_set_ratio handles word reordering; partial_ratio handles substrings
    matches = fuzz_process.extractBests(
        query,
        name_strings,
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_THRESHOLD,
        limit=10
    )

    if not matches:
        return []

    results = []
    seen_indices = set()

    for matched_name, score in matches:
        # Find the original entry for this matched name string
        for name_str, df_idx, field in name_list:
            if name_str == matched_name and df_idx not in seen_indices:
                medicine = df.iloc[df_idx].to_dict()
                results.append({
                    "medicine": medicine,
                    "score": score,
                    "matched_on": field
                })
                seen_indices.add(df_idx)
                break

    # Sort by score descending
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
