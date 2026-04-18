"""
setup.py
========
One-time setup script for MedScan AI. Run this before launching the app:

    python setup.py

What this script does:
  1. Creates required directories (models/, data/, sample_images/) if missing
  2. Downloads NLTK resources (punkt tokeniser, stopwords, wordnet) to local cache
  3. Verifies that data/medicines.csv and data/intents.json exist
  4. Loads the medicine database and validates its schema
  5. Trains the Naive Bayes intent classifier and saves it to models/
  6. Builds the TF-IDF medicine search index and saves it to models/

Why a separate setup step?
  Training ML models every time the app starts would add 2-5 seconds to
  every launch. By saving the trained models to disk with joblib, subsequent
  launches load in under 0.5 seconds.

After running setup.py successfully, launch the app with:
    streamlit run app.py

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import os
import sys
import logging
import time

# ─────────────────────────────────────────────────────────────────────
# PATH SETUP — ensure imports resolve from project root
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("setup")


# ─────────────────────────────────────────────────────────────────────
# DIRECTORY CREATION
# ─────────────────────────────────────────────────────────────────────

def create_directories() -> None:
    """Create all required project directories if they don't exist."""
    dirs = [
        os.path.join(PROJECT_ROOT, "models"),
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "data", "sample_images"),
        os.path.join(PROJECT_ROOT, "docs"),
        os.path.join(PROJECT_ROOT, "tests"),
        os.path.join(PROJECT_ROOT, "components"),
        os.path.join(PROJECT_ROOT, "modules"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Directories created")


# ─────────────────────────────────────────────────────────────────────
# MAIN SETUP FUNCTION
# ─────────────────────────────────────────────────────────────────────

def main():
    """Run the complete setup pipeline."""
    print("\n" + "═" * 60)
    print("  MEDSCAN AI — Setup Script")
    print("  MedScan AI Project — Offline Medicine Intelligence System")
    print("═" * 60)
    print()

    start_time = time.perf_counter()

    # ── Step 1: Create directories ─────────────────────────────────────
    print("[1/6] Creating required directories...")
    create_directories()

    # ── Step 2: Download NLTK resources ───────────────────────────────
    print("[2/6] Downloading NLTK resources (punkt, stopwords, wordnet)...")
    try:
        from modules.preprocessor import download_nltk_resources
        download_nltk_resources()
        print("✓ NLTK resources ready")
    except Exception as e:
        print(f"  ⚠ NLTK download warning (non-fatal): {e}")

    # ── Step 3: Verify data files ──────────────────────────────────────
    print("[3/6] Verifying data files...")
    from modules.medicine_db import DB_PATH
    intents_path = os.path.join(PROJECT_ROOT, "data", "intents.json")

    if not os.path.exists(DB_PATH):
        print(f"  ✗ Medicine database not found at {DB_PATH}")
        sys.exit(1)

    if not os.path.exists(intents_path):
        print(f"  ✗ intents.json not found at {intents_path}")
        sys.exit(1)

    print("✓ Data files verified")

    # ── Step 4: Load medicine database ────────────────────────────────
    print("[4/6] Loading medicine database...")
    try:
        from modules.medicine_db import load_database
        df = load_database(DB_PATH)
        print(f"✓ Database loaded: {len(df)} medicines")
    except Exception as e:
        print(f"  ✗ Database load failed: {e}")
        sys.exit(1)

    # ── Step 5: Train Naive Bayes intent classifier ───────────────────
    print("[5/6] Training Naive Bayes intent classifier...")
    try:
        from modules.chatbot import load_intents
        from modules.intent_classifier import prepare_training_data, train_classifier, save_models

        intents = load_intents(intents_path)
        X, y = prepare_training_data(intents)
        vectorizer, classifier, label_encoder = train_classifier(X, y)
        save_models(vectorizer, classifier, label_encoder)

        from sklearn.metrics import accuracy_score
        X_vec = vectorizer.transform(X)
        y_enc = label_encoder.transform(y)
        preds = classifier.predict(X_vec)
        acc = accuracy_score(y_enc, preds)
        print(f"✓ Naive Bayes trained (training accuracy: {acc*100:.1f}%)")

    except Exception as e:
        print(f"  ✗ Classifier training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 6: Build TF-IDF search index ─────────────────────────────
    print("[6/6] Building TF-IDF medicine search index...")
    try:
        import joblib
        from modules.medicine_db import build_search_corpus
        from modules.medicine_search import build_search_index

        corpus = build_search_corpus(df)
        search_vec, search_mat = build_search_index(corpus)

        models_dir = os.path.join(PROJECT_ROOT, "models")
        search_index_path = os.path.join(models_dir, "search_index.pkl")
        joblib.dump((search_vec, search_mat), search_index_path)
        print(f"✓ Search index built and saved ({search_mat.shape[0]} docs, {search_mat.shape[1]} features)")

    except Exception as e:
        print(f"  ✗ Search index build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Setup complete ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    print()
    print("═" * 60)
    print(f"  ✓ Setup complete in {elapsed:.1f}s")
    print()
    print("  TO LAUNCH MEDSCAN AI:")
    print("  streamlit run app.py")
    print()
    print("═" * 60)
    print()

if __name__ == "__main__":
    main()
