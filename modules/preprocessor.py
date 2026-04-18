"""
preprocessor.py
===============
This module handles all text preprocessing for the MedScan AI chatbot and
search engine. Before any machine learning can happen, raw user input needs
to be cleaned, tokenised, and filtered down to its meaningful content.

Why preprocessing matters here:
  - A user might type "What ARE the Side-Effects??", "what r side effects",
    or "SIDE EFFECTS?" — all should map to the same intent.
  - Common English words like "what", "are", "the" carry no meaning for
    intent classification and should be removed.
  - Medical terms like "pregnancy", "dosage", "overdose" must be preserved
    even if they would normally be treated as stopwords.

Pipeline applied to every piece of text:
  1. clean_text        → lowercase, remove punctuation and special characters
  2. tokenize          → split sentence into individual word tokens (NLTK punkt)
  3. remove_stopwords  → filter out common English words (but keep medical ones)
  4. stem_tokens       → (optional) reduce words to root form e.g. "dosing"→"dose"
  5. rejoin            → return as a single string ready for TF-IDF vectorisation

All NLTK resources are downloaded once by download_nltk_resources() during setup.
At runtime, everything runs from the local cache — no internet needed.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""

import re
import string
import logging

# Third-party NLP library — must be available after `pip install nltk`
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Medical terms that should NOT be removed as stopwords
# These words carry clinical meaning and improve intent matching
MEDICAL_TERMS_WHITELIST = {
    "pain", "dose", "drug", "side", "effect", "use", "take", "safe",
    "child", "adult", "pregnant", "pregnancy", "liver", "kidney",
    "blood", "heart", "eye", "skin", "tablet", "capsule", "syrup",
    "injection", "cream", "gel", "drop", "dosage", "warning",
    "danger", "interaction", "substitute", "alternative", "overdose",
    "fever", "infection", "allergy", "antibiotic", "steroid",
    "reaction", "treatment", "medicine", "medical", "doctor"
}

# Singleton stemmer instance (avoid re-creating per call)
_stemmer = None

# Module-level logger
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# NLTK RESOURCE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

def download_nltk_resources() -> None:
    """
    Download all required NLTK corpus resources silently.

    Downloads: punkt (tokenizer), stopwords (word list),
               wordnet (WordNet lemmatiser), averaged_perceptron_tagger (POS).
    Safe to call multiple times — NLTK skips already-downloaded resources.

    Raises:
        Exception: Logged as warning if any download fails (non-fatal for most functions).

    Example:
        >>> download_nltk_resources()
        # No output on success
    """
    # List of (resource_name, resource_path) tuples to check/download
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]

    for path, name in resources:
        try:
            # Attempt to find resource — triggers download if missing
            nltk.data.find(path)
        except LookupError:
            # Resource not found locally — download silently
            try:
                nltk.download(name, quiet=True)
                logger.info(f"NLTK resource downloaded: {name}")
            except Exception as e:
                # Log but don't crash — app can still function without some resources
                logger.warning(f"Could not download NLTK resource '{name}': {e}")


# ─────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase and remove special characters from raw text input.

    Removes all non-alphanumeric characters except ASCII apostrophes
    (preserved for contractions like "don't", "can't").
    Multiple whitespace is collapsed to a single space.

    Args:
        text (str): Raw input string to clean.

    Returns:
        str: Cleaned, lowercased string with normalised whitespace.

    Example:
        >>> clean_text("What ARE the Side-Effects?!")
        'what are the side effects'
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Lowercase all characters for uniform comparison
    text = text.lower()

    # Step 2: Keep alphanumeric + spaces + apostrophes (contractions)
    # Remove all other special characters including hyphens, punctuation
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Step 3: Collapse multiple consecutive whitespace into one space
    text = re.sub(r"\s+", " ", text)

    # Step 4: Strip leading/trailing whitespace
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """
    Tokenise input text into a list of word tokens using NLTK word_tokenize.

    Falls back to simple whitespace split if NLTK punkt tokenizer
    is not available (graceful degradation).

    Args:
        text (str): Input string to tokenise.

    Returns:
        list[str]: List of word tokens.

    Example:
        >>> tokenize("what are the side effects")
        ['what', 'are', 'the', 'side', 'effects']
    """
    if not text:
        return []

    try:
        # NLTK word tokenizer: handles contractions, punctuation, edge cases
        return word_tokenize(text)
    except LookupError:
        # Fallback if punkt data not available: simple split on whitespace
        logger.warning("NLTK punkt not available — falling back to split()")
        return text.split()


# ─────────────────────────────────────────────────────────────────────
# STOPWORD REMOVAL
# ─────────────────────────────────────────────────────────────────────

def remove_stopwords(tokens: list) -> list:
    """
    Remove English stopwords from a token list while preserving medical terms.

    Uses NLTK's English stopwords list minus the MEDICAL_TERMS_WHITELIST
    to ensure clinically meaningful words are kept for intent classification.

    Args:
        tokens (list[str]): List of word tokens.

    Returns:
        list[str]: Filtered token list with stopwords removed.

    Example:
        >>> remove_stopwords(['what', 'are', 'the', 'side', 'effects'])
        ['side', 'effects']
    """
    if not tokens:
        return []

    try:
        # Load NLTK English stopwords set
        stop_words = set(stopwords.words("english"))

        # Remove medical whitelisted terms from the stopword set
        # so they are not inadvertently removed during filtering
        stop_words -= MEDICAL_TERMS_WHITELIST

        # Keep only tokens that are NOT in the filtered stopword set
        return [t for t in tokens if t.lower() not in stop_words]

    except LookupError:
        # NLTK stopwords not available — return tokens unchanged
        logger.warning("NLTK stopwords not available — skipping stopword removal")
        return tokens


# ─────────────────────────────────────────────────────────────────────
# STEMMING
# ─────────────────────────────────────────────────────────────────────

def stem_tokens(tokens: list) -> list:
    """
    Apply Porter stemming to reduce tokens to their root forms.

    Example: ["dosages", "dosing"] → ["dosag", "dose"] (approximate)
    Note: stemming is lossy — use only when exact form is unimportant.

    Args:
        tokens (list[str]): List of word tokens to stem.

    Returns:
        list[str]: List of stemmed tokens.

    Example:
        >>> stem_tokens(['running', 'effects', 'dosage'])
        ['run', 'effect', 'dosag']
    """
    global _stemmer

    # Lazy initialisation: create stemmer only on first call
    if _stemmer is None:
        _stemmer = PorterStemmer()

    # Apply stem to each token; skip empty strings
    return [_stemmer.stem(t) for t in tokens if t]


# ─────────────────────────────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────

def preprocess_pipeline(text: str, stem: bool = False) -> str:
    """
    Run the complete NLP preprocessing pipeline on input text.

    Pipeline steps:
      1. clean_text      — lowercase, remove special chars
      2. tokenize        — split into word tokens
      3. remove_stopwords— filter unhelpful common words
      4. (optional) stem — reduce to root forms
      5. rejoin          — return as space-joined string

    Args:
        text (str): Raw input text string.
        stem (bool): If True, apply Porter stemming. Default False.
                     Stemming is useful for search indexing but may
                     reduce readability of reconstructed text.

    Returns:
        str: Preprocessed text as a single joined string.

    Example:
        >>> preprocess_pipeline("What are the side effects of this medicine?")
        'side effects medicine'
    """
    if not text:
        return ""

    # Step 1: Clean and normalise raw text
    cleaned = clean_text(text)

    # Step 2: Break into individual word tokens
    tokens = tokenize(cleaned)

    # Step 3: Remove non-informative stopwords (preserve medical terms)
    tokens = remove_stopwords(tokens)

    # Step 4 (optional): Stem tokens for more aggressive normalisation
    if stem:
        tokens = stem_tokens(tokens)

    # Step 5: Rejoin into a single space-separated string for vectoriser input
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────
# MEDICAL KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_medical_keywords(text: str) -> list:
    """
    Identify and extract medical domain keywords from input text.

    Checks each token in the cleaned text against the MEDICAL_TERMS_WHITELIST.
    Useful for highlighting clinically relevant words in chatbot input.

    Args:
        text (str): Raw input text string.

    Returns:
        list[str]: Unique list of recognised medical keywords found in text.

    Example:
        >>> extract_medical_keywords("Is this safe during pregnancy?")
        ['safe', 'pregnancy']
    """
    if not text:
        return []

    # Clean and tokenise the input text
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    # Return deduplicated list of tokens that appear in the medical whitelist
    found = [t for t in tokens if t.lower() in MEDICAL_TERMS_WHITELIST]

    # Remove duplicates while preserving first-occurrence order
    seen = set()
    unique_found = []
    for word in found:
        if word not in seen:
            seen.add(word)
            unique_found.append(word)

    return unique_found
