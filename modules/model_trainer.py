"""
model_trainer.py
================
This module orchestrates loading and saving of ML model artifacts.

Role in the system:
  At startup (both via setup.py and app.py), the app needs:
  - The TF-IDF search index (for medicine name lookup)
  - The Naive Bayes intent classifier (for chatbot)
  - The TF-IDF intent vectoriser (for chatbot pattern matching)

  This module decides whether to load pre-saved models from models/
  (fast path, used on every launch after setup) or train them fresh
  (slow path, used only when models/ is empty or corrupted).

  Using joblib for model persistence is standard practice with scikit-learn.
  It serialises the Python objects efficiently to .pkl files.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import os
import logging
import joblib

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SEARCH_INDEX_PATH = os.path.join(MODEL_DIR, "search_index.pkl")


# ─────────────────────────────────────────────────────────────────────
# SEARCH INDEX TRAINING
# ─────────────────────────────────────────────────────────────────────

def train_and_save_search_index(df) -> tuple:
    """
    Build the TF-IDF medicine search index and save to disk.

    Args:
        df (pd.DataFrame): Loaded medicine DataFrame.

    Returns:
        tuple: (vectorizer, tfidf_matrix) for the search index.
    """
    from modules.medicine_db import build_search_corpus
    from modules.medicine_search import build_search_index

    # Build text corpus (one doc per medicine, combining all searchable fields)
    corpus = build_search_corpus(df)

    # Fit TF-IDF vectoriser and transform corpus
    vectorizer, search_matrix = build_search_index(corpus)

    # Save both components together as a tuple
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump((vectorizer, search_matrix), SEARCH_INDEX_PATH)

    logger.info(f"Search index saved to {SEARCH_INDEX_PATH}")
    return vectorizer, search_matrix


def load_search_index() -> tuple:
    """
    Load saved search index from disk if available.

    Returns:
        tuple | None: (vectorizer, matrix) or None if not found.
    """
    if not os.path.exists(SEARCH_INDEX_PATH):
        return None
    try:
        result = joblib.load(SEARCH_INDEX_PATH)
        logger.info("Search index loaded from disk")
        return result
    except Exception as e:
        logger.warning(f"Could not load search index: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# UNIFIED TRAINING
# ─────────────────────────────────────────────────────────────────────

def run_full_training(force_retrain: bool = False) -> dict:
    """
    Run complete model training pipeline:
      1. Load medicine database
      2. Train/load search index
      3. Train/load Naive Bayes intent classifier
      4. Build/load chatbot TF-IDF intent index

    Args:
        force_retrain (bool): If True, retrain even if saved models exist.

    Returns:
        dict: All trained model components:
            {
              "search_vectorizer": TfidfVectorizer,
              "search_matrix": sparse matrix,
              "nb_vectorizer": TfidfVectorizer,
              "nb_classifier": MultinomialNB,
              "label_encoder": LabelEncoder,
              "chat_vectorizer": TfidfVectorizer,
              "chat_matrix": sparse matrix,
              "chat_intent_labels": list[str],
              "medicine_df": pd.DataFrame
            }
    """
    from modules.medicine_db import load_database
    from modules.intent_classifier import init_classifier, VECTORIZER_PATH
    from modules.chatbot import load_intents, build_intent_index

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load medicine database ────────────────────────────────────────
    logger.info("Loading medicine database...")
    df = load_database()

    # ── Search index ──────────────────────────────────────────────────
    if not force_retrain:
        search_result = load_search_index()
    else:
        search_result = None

    if search_result is None:
        logger.info("Building search index...")
        search_vec, search_mat = train_and_save_search_index(df)
    else:
        search_vec, search_mat = search_result

    # ── Naive Bayes intent classifier ─────────────────────────────────
    logger.info("Initialising Naive Bayes intent classifier...")
    nb_vec, nb_clf, nb_enc = init_classifier()

    # ── Chatbot TF-IDF index ──────────────────────────────────────────
    logger.info("Building chatbot intent index...")
    intents = load_intents()
    chat_vec, chat_mat, chat_labels = build_intent_index(intents)

    logger.info("All models ready.")

    return {
        "search_vectorizer": search_vec,
        "search_matrix": search_mat,
        "nb_vectorizer": nb_vec,
        "nb_classifier": nb_clf,
        "label_encoder": nb_enc,
        "chat_vectorizer": chat_vec,
        "chat_matrix": chat_mat,
        "chat_intent_labels": chat_labels,
        "medicine_df": df,
    }
