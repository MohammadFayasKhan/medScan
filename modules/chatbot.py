"""
chatbot.py
==========
This module powers the offline MedBot chatbot that answers clinical
questions about the currently displayed medicine.

How the chatbot works:
  1. The user types a question (e.g. "Is it safe during pregnancy?")
  2. The question is preprocessed (lowercase, remove stopwords)
  3. A TF-IDF vectoriser converts it to a numeric feature vector
  4. Cosine similarity is computed against all 799 example patterns
     from intents.json (14 categories × ~57 patterns each)
  5. The best matching intent is identified (e.g. "pregnancy")
  6. A response template for that intent is selected and filled in
     with the specific medicine's data fields from the database

Why TF-IDF + cosine similarity instead of a large language model?
  - Fully offline — no GPT/API calls needed
  - Deterministic and explainable — we can trace why a response was given
  - Fast — classification completes in milliseconds
  - Accurate enough — 99.1% training accuracy on medical intent patterns

The Naive Bayes classifier (intent_classifier.py) is also used as a
secondary classifier for cross-validation of intent predictions.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import json
import time
import os
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.preprocessor import preprocess_pipeline, extract_medical_keywords

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

INTENTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "intents.json"
)

# Minimum cosine similarity score to accept an intent match
# Below this → classify as "unknown" and show help message
MIN_CONFIDENCE = 0.08

# TF-IDF n-gram range for intent pattern vectorisation
NGRAM_RANGE = (1, 2)

# Emoji icons per intent for response formatting
INTENT_ICONS = {
    "overview": "💊",
    "uses": "🎯",
    "mechanism": "⚙️",
    "dosage": "💉",
    "side_effects": "⚠️",
    "warnings": "🚨",
    "contraindications": "🚫",
    "interactions": "🔗",
    "substitutes": "🔄",
    "pregnancy": "🤰",
    "pediatric": "👶",
    "admin_tips": "📋",
    "storage": "🏪",
    "overdose": "🆘",
    "unknown": "❓",
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# INTENT LOADING
# ─────────────────────────────────────────────────────────────────────

def load_intents(path: str = INTENTS_PATH) -> dict:
    """
    Load chatbot intents from the local JSON file.

    Validates that the file exists and has the correct structure.
    Does not require internet — purely local file read.

    Args:
        path (str): Absolute or relative path to intents.json.

    Returns:
        dict: Parsed intents dictionary with 'intents' key.

    Raises:
        FileNotFoundError: If intents.json is not found at path.
        ValueError: If JSON structure is missing 'intents' key.

    Example:
        >>> intents = load_intents()
        >>> len(intents["intents"]) == 14
        True
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"intents.json not found at: {path}\n"
            "Run setup.py to ensure all data files are present."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if "intents" not in data:
        raise ValueError("intents.json is missing the required 'intents' key.")

    logger.info(f"Intents loaded: {len(data['intents'])} categories")
    return data


# ─────────────────────────────────────────────────────────────────────
# INTENT INDEX BUILDING
# ─────────────────────────────────────────────────────────────────────

def build_intent_index(intents: dict) -> tuple:
    """
    Build a TF-IDF index from all intent patterns for similarity matching.

    Each question pattern from intents.json becomes one document.
    Multiple patterns from the same intent all map to the same intent tag.
    
    (vectorizer, matrix) are then used in classify_intent() to match
    real user questions to the closest intent category.

    Args:
        intents (dict): Loaded intents dict (from load_intents()).

    Returns:
        tuple[TfidfVectorizer, sparse_matrix, list[str]]:
            (fitted_vectorizer, tfidf_pattern_matrix, intent_labels)
            intent_labels[i] = intent tag for pattern at matrix row i.

    Example:
        >>> vec, mat, labels = build_intent_index(intents)
        >>> len(labels) > 100  # > 100 patterns total
        True
    """
    docs = []          # flat list of all pattern strings
    intent_labels = [] # corresponding intent tag for each pattern

    for intent in intents["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            # Preprocess each pattern for consistent vectorisation
            cleaned = preprocess_pipeline(pattern, stem=False)
            docs.append(cleaned if cleaned else pattern.lower())
            intent_labels.append(tag)

    # Build TF-IDF vectoriser with bigrams for better phrase matching
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        strip_accents="unicode",
        analyzer="word",
        min_df=1,
        max_features=5000
    )

    # Fit and transform all patterns into TF-IDF matrix
    # Shape: (num_patterns, vocabulary_size)
    tfidf_matrix = vectorizer.fit_transform(docs)

    logger.info(f"Intent index built: {len(docs)} patterns, {len(set(intent_labels))} intents")
    return vectorizer, tfidf_matrix, intent_labels


# ─────────────────────────────────────────────────────────────────────
# INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────

def preprocess_question(question: str) -> str:
    """
    Clean and normalise a user question for better intent matching.

    Applies the full NLP preprocessing pipeline to remove noise while
    preserving medically relevant keywords.

    Args:
        question (str): Raw user-typed question string.

    Returns:
        str: Preprocessed question for TF-IDF vectorisation.

    Example:
        >>> preprocess_question("What are the side effects?")
        'side effects'
    """
    return preprocess_pipeline(question, stem=False)


def classify_intent(question: str, vectorizer,
                    matrix, intent_labels: list) -> tuple:
    """
    Classify a user question into the most similar intent category.

    Algorithm:
      1. Preprocess question (lowercase, remove stopwords)
      2. Transform into TF-IDF vector
      3. Compute cosine similarity vs all intent patterns in matrix
      4. Find argmax (highest similarity pattern)
      5. If score < MIN_CONFIDENCE → return "unknown"
      6. Else → return intent tag for that pattern

    Args:
        question (str): Raw user question string.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectoriser from build_intent_index.
        matrix: TF-IDF pattern matrix from build_intent_index.
        intent_labels (list[str]): Intent tag for each row in matrix.

    Returns:
        tuple[str, float]: (intent_tag, confidence_score).
            intent_tag: one of the 14 intent categories or "unknown".
            confidence_score: 0.0–1.0 cosine similarity value.

    Example:
        >>> tag, conf = classify_intent("what are side effects", vec, mat, labels)
        >>> tag
        'side_effects'
        >>> conf > 0.3
        True
    """
    if not question or vectorizer is None or matrix is None:
        return ("unknown", 0.0)

    # Step 1: Preprocess question the same way patterns were processed
    processed_q = preprocess_question(question)
    if not processed_q:
        # Fallback: use whole original question if preprocessing removes everything
        processed_q = question.lower()

    # Step 2: Transform question into TF-IDF vector
    try:
        q_vector = vectorizer.transform([processed_q])
    except Exception:
        return ("unknown", 0.0)

    # Step 3: Cosine similarity between question vector and all patterns
    similarities = cosine_similarity(q_vector, matrix).flatten()

    # Step 4: Find index of highest similarity score
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    # Step 5: Confidence threshold — below MIN_CONFIDENCE = unrecognised
    if best_score < MIN_CONFIDENCE:
        logger.debug(f"Intent below threshold: '{question}' score={best_score:.3f}")
        return ("unknown", 0.0)

    # Step 6: Return the intent tag corresponding to the best matching pattern
    intent_tag = intent_labels[best_idx]
    logger.debug(f"Intent classified: '{intent_tag}' (confidence={best_score:.3f})")
    return (intent_tag, best_score)


# ─────────────────────────────────────────────────────────────────────
# RESPONSE GENERATION
# ─────────────────────────────────────────────────────────────────────

def _format_list(items, bullet: str = "•") -> str:
    """Helper: format a list of items as a markdown bullet list string."""
    if not items:
        return "*Not specified*"
    if isinstance(items, str):
        items = [i.strip() for i in items.split(",") if i.strip()]
    return "\n".join(f"{bullet} {item}" for item in items if item)


def generate_response(intent_tag: str, medicine: dict, confidence: float) -> str:
    """
    Generate a dynamic, contextual response from the medicine data dict.

    Each intent maps to specific fields in the medicine dict.
    All response content comes from medicine data — nothing is hardcoded.
    Responses are formatted in Markdown for Streamlit rendering.

    Args:
        intent_tag (str): Classified intent from classify_intent().
        medicine (dict): Full medicine data dict from the database.
        confidence (float): Confidence score from classifier (0.0–1.0).

    Returns:
        str: Markdown-formatted response string.

    Example:
        >>> resp = generate_response("uses", med_dict, 0.85)
        >>> "used for" in resp.lower()
        True
    """
    name = medicine.get("name", "this medicine").title()
    generic = medicine.get("generic_name", "")
    icon = INTENT_ICONS.get(intent_tag, "💊")

    # Low-confidence disclaimer appended to response
    low_conf_note = (
        "\n\n---\n*⚠️ Low confidence match. If this doesn't answer your question, "
        "try rephrasing.*"
        if confidence < 0.3 else ""
    )

    # ── Intent response map ──────────────────────────────────────────
    if intent_tag == "overview":
        features = medicine.get("features", [])
        features_str = _format_list(features)
        return (
            f"{icon} **Overview: {name}**\n\n"
            f"**Generic Name:** {generic}\n\n"
            f"**Category:** {medicine.get('category', 'Not specified')}\n\n"
            f"**Form:** {medicine.get('form', 'Not specified')}\n\n"
            f"**Strength:** {medicine.get('strength', 'Not specified')}\n\n"
            f"**Active Substance:** {medicine.get('active_substance', 'Not specified')}\n\n"
            f"**Key Features:**\n{features_str}"
            + low_conf_note
        )

    elif intent_tag == "uses":
        indications = medicine.get("indications", [])
        return (
            f"{icon} **Uses of {name}**\n\n"
            f"{medicine.get('uses', 'Not specified')}\n\n"
            f"**Medical Indications:**\n{_format_list(indications)}"
            + low_conf_note
        )

    elif intent_tag == "mechanism":
        return (
            f"{icon} **How {name} Works**\n\n"
            f"{medicine.get('mechanism', 'Mechanism information not specified.')}"
            + low_conf_note
        )

    elif intent_tag == "dosage":
        return (
            f"{icon} **Dosage & Administration of {name}**\n\n"
            f"**Recommended Dose:**\n{medicine.get('dosage', 'Not specified')}\n\n"
            f"**When to Take:** {medicine.get('timing', 'Not specified')}\n\n"
            f"**Dosing Interval:** {medicine.get('spacing', 'Not specified')}\n\n"
            f"**Administration Tips:** {medicine.get('admin_tips', 'Not specified')}"
            + low_conf_note
        )

    elif intent_tag == "side_effects":
        common = medicine.get("side_effects_common", [])
        serious = medicine.get("side_effects_serious", [])
        return (
            f"{icon} **Side Effects of {name}**\n\n"
            f"**Common Side Effects:**\n{_format_list(common, '🟡')}\n\n"
            f"**Serious Side Effects (seek medical attention):**\n{_format_list(serious, '🔴')}"
            + low_conf_note
        )

    elif intent_tag == "warnings":
        return (
            f"{icon} **Warnings & Precautions for {name}**\n\n"
            f"**🤰 Pregnancy:** {medicine.get('warning_pregnancy', 'Not specified')}\n\n"
            f"**👶 Pediatric Use:** {medicine.get('warning_pediatric', 'Not specified')}\n\n"
            f"**🚗 Driving:** {medicine.get('warning_driving', 'Not specified')}\n\n"
            f"**🏪 Storage:** {medicine.get('warning_storage', 'Not specified')}"
            + low_conf_note
        )

    elif intent_tag == "contraindications":
        contra = medicine.get("contraindications", [])
        return (
            f"{icon} **Contraindications for {name}**\n\n"
            f"**Do NOT use {name} if you have:**\n{_format_list(contra, '❌')}\n\n"
            f"*Always consult your doctor before stopping any medication.*"
            + low_conf_note
        )

    elif intent_tag == "interactions":
        interactions = medicine.get("interactions", [])
        return (
            f"{icon} **Drug Interactions of {name}**\n\n"
            f"**Known Interactions:**\n{_format_list(interactions, '⚡')}\n\n"
            f"*Inform your doctor about all medicines, supplements, and foods you consume.*"
            + low_conf_note
        )

    elif intent_tag == "substitutes":
        substitutes = medicine.get("substitutes", [])
        return (
            f"{icon} **Alternatives to {name}**\n\n"
            f"**Possible Substitutes:**\n{_format_list(substitutes, '🔄')}\n\n"
            f"⚠️ *Disclaimer: Do not switch medications without consulting your doctor or pharmacist. "
            f"Therapeutic equivalence varies by patient condition.*"
            + low_conf_note
        )

    elif intent_tag == "pregnancy":
        return (
            f"{icon} **{name} During Pregnancy & Breastfeeding**\n\n"
            f"**Pregnancy Safety:**\n{medicine.get('warning_pregnancy', 'Not specified')}\n\n"
            f"*Always consult your obstetrician or GP before taking any medicine during pregnancy.*"
            + low_conf_note
        )

    elif intent_tag == "pediatric":
        return (
            f"{icon} **{name} in Children**\n\n"
            f"**Pediatric Use:**\n{medicine.get('warning_pediatric', 'Not specified')}\n\n"
            f"*Always consult a pediatrician for appropriate dosing in children.*"
            + low_conf_note
        )

    elif intent_tag == "admin_tips":
        return (
            f"{icon} **How to Take {name}**\n\n"
            f"**Administration Instructions:**\n{medicine.get('admin_tips', 'Not specified')}\n\n"
            f"**Timing:** {medicine.get('timing', 'Not specified')}\n\n"
            f"**Dosing Interval:** {medicine.get('spacing', 'Not specified')}"
            + low_conf_note
        )

    elif intent_tag == "storage":
        return (
            f"{icon} **Storage Instructions for {name}**\n\n"
            f"{medicine.get('warning_storage', 'Not specified')}"
            + low_conf_note
        )

    elif intent_tag == "overdose":
        dosage = medicine.get("dosage", "")
        return (
            f"{icon} **OVERDOSE — {name}**\n\n"
            f"**⚠️ If you have taken more than the prescribed amount:**\n\n"
            f"1. **Stop taking the medicine immediately**\n"
            f"2. **Call emergency services (108/112) or go to your nearest hospital**\n"
            f"3. **Do NOT try to induce vomiting unless specifically advised**\n\n"
            f"**Prescribed Dose for Reference:**\n{dosage}\n\n"
            f"🆘 *This is a medical emergency. Contact Poison Control or emergency services immediately.*"
            + low_conf_note
        )

    else:
        # Unknown intent — show helpful categories list
        return (
            f"❓ **I'm not sure about that question.**\n\n"
            f"I can answer questions about **{name}** in these areas:\n\n"
            "• 💊 **Overview** — what type of medicine this is\n"
            "• 🎯 **Uses** — what conditions it treats\n"
            "• ⚙️ **Mechanism** — how it works in the body\n"
            "• 💉 **Dosage** — how much and how often\n"
            "• ⚠️ **Side Effects** — common and serious reactions\n"
            "• 🚨 **Warnings** — pregnancy, children, driving\n"
            "• 🚫 **Contraindications** — who cannot take this\n"
            "• 🔗 **Interactions** — drug and food interactions\n"
            "• 🔄 **Substitutes** — alternative medicines\n"
            "• 📋 **How to Take** — administration instructions\n"
            "• 🏪 **Storage** — how to store this medicine\n"
            "• 🆘 **Overdose** — what to do if too much taken\n\n"
            "*Try asking: 'What are the side effects?' or 'How should I take this?'*"
        )


# ─────────────────────────────────────────────────────────────────────
# MASTER CHATBOT FUNCTION
# ─────────────────────────────────────────────────────────────────────

def get_chat_response(user_input: str, medicine: dict,
                      vectorizer, matrix, intent_labels: list) -> dict:
    """
    Master chatbot entry point — called by chatbot_ui.py on every message.

    Orchestrates intent classification and response generation.
    Returns a structured result dict with all metadata for UI rendering.

    Args:
        user_input (str): Raw user question string.
        medicine (dict): Currently displayed medicine data dict.
        vectorizer: Fitted TF-IDF vectoriser from build_intent_index.
        matrix: TF-IDF pattern matrix from build_intent_index.
        intent_labels (list[str]): Intent labels list from build_intent_index.

    Returns:
        dict: Chat response with keys:
            {
              "response": str,           # Markdown-formatted response text
              "intent": str,             # Detected intent tag
              "confidence": float,       # 0.0–1.0 similarity score
              "low_confidence": bool,    # True if confidence < 0.3
              "response_time_ms": float  # Processing time in milliseconds
            }

    Example:
        >>> result = get_chat_response("what are side effects", med, vec, mat, labels)
        >>> result["intent"]
        'side_effects'
    """
    start_time = time.perf_counter()

    # Handle empty or whitespace-only input
    if not user_input or not user_input.strip():
        return {
            "response": "Please type a question about the medicine.",
            "intent": "unknown",
            "confidence": 0.0,
            "low_confidence": True,
            "response_time_ms": 0.0
        }

    # Handle case where no medicine is currently selected
    if not medicine:
        return {
            "response": (
                "⚠️ **No medicine selected.**\n\n"
                "Please scan or search for a medicine first, "
                "then ask questions about it here."
            ),
            "intent": "unknown",
            "confidence": 0.0,
            "low_confidence": True,
            "response_time_ms": 0.0
        }

    # ── Step 1: Classify user question into an intent ────────────────
    intent_tag, confidence = classify_intent(
        user_input, vectorizer, matrix, intent_labels
    )

    # ── Step 2: Generate contextual response from medicine data ───────
    response_text = generate_response(intent_tag, medicine, confidence)

    # ── Step 3: Calculate total processing time ───────────────────────
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"Chat response: intent='{intent_tag}', "
        f"confidence={confidence:.3f}, time={elapsed_ms:.1f}ms"
    )

    return {
        "response": response_text,
        "intent": intent_tag,
        "confidence": confidence,
        "low_confidence": confidence < 0.3,
        "response_time_ms": elapsed_ms
    }
