"""
setup.py
========
MEDSCAN AI — One-time Setup Script.

Run this ONCE before launching app.py for the first time.

Steps:
  1. Create required directories (models/, data/, sample_images/)
  2. Download NLTK resources (punkt, stopwords, wordnet, tagger)
  3. Load intents.json
  4. Train Naive Bayes intent classifier → save to models/
  5. Load medicine database
  6. Build TF-IDF search index → save to models/
  7. Print setup complete message with launch instructions

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
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
    print("  ANTIGRAVITY BUILD — Offline Medicine Intelligence System")
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
    db_path = os.path.join(PROJECT_ROOT, "data", "medicines.csv")
    intents_path = os.path.join(PROJECT_ROOT, "data", "intents.json")

    if not os.path.exists(db_path):
        print(f"  ✗ medicines.csv not found at {db_path}")
        print("    Please ensure data/medicines.csv exists. See DATASET_GUIDE.md.")
        sys.exit(1)

    if not os.path.exists(intents_path):
        print(f"  ✗ intents.json not found at {intents_path}")
        sys.exit(1)

    print("✓ Data files verified")

    # ── Step 4: Load medicine database ────────────────────────────────
    print("[4/6] Loading medicine database...")
    try:
        import pandas as pd
        from modules.medicine_db import load_database, build_search_corpus

        # Bypass Streamlit cache for setup script
        df = pd.read_csv(db_path, encoding="utf-8")
        df = df.fillna("Not specified")
        for col in ["brand_names", "features", "contraindications", "interactions",
                    "side_effects_common", "side_effects_serious", "substitutes",
                    "pack_sizes", "sources", "indications"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: [item.strip() for item in str(x).split(",")]
                    if str(x) not in ("Not specified", "nan") else []
                )
        df["name_lower"] = df["name"].str.lower().str.strip()
        print(f"✓ Database loaded: {len(df)} medicines")
    except Exception as e:
        print(f"  ✗ Database load failed: {e}")
        sys.exit(1)

    # ── Step 5: Train Naive Bayes intent classifier ───────────────────
    print("[5/6] Training Naive Bayes intent classifier...")
    try:
        from modules.chatbot import load_intents
        from modules.intent_classifier import (
            prepare_training_data, train_classifier, save_models
        )

        intents = load_intents(intents_path)
        X, y = prepare_training_data(intents)
        vectorizer, classifier, label_encoder = train_classifier(X, y)
        save_models(vectorizer, classifier, label_encoder)

        # Quick accuracy report on training data
        from sklearn.metrics import accuracy_score
        X_vec = vectorizer.transform(X)
        from sklearn.preprocessing import LabelEncoder
        le = label_encoder
        y_enc = le.transform(y)
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

        # Build combined text corpus from medicine fields
        list_cols = ["brand_names", "features", "indications", "substitutes"]
        corpus = []
        for _, row in df.iterrows():
            name_boost = f"{row['name']} {row['name']} {row['name']}"
            brands = " ".join(row.get("brand_names", [])) if isinstance(row.get("brand_names"), list) else str(row.get("brand_names", ""))
            indications = " ".join(row.get("indications", [])) if isinstance(row.get("indications"), list) else str(row.get("indications", ""))
            doc = " ".join([
                name_boost,
                str(row.get("generic_name", "")),
                brands,
                str(row.get("active_substance", "")),
                str(row.get("category", "")),
                str(row.get("uses", "")),
                indications,
            ])
            corpus.append(doc)

        from modules.medicine_search import build_search_index
        search_vec, search_mat = build_search_index(corpus)

        # Save search index
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
    print("  App will open at: http://localhost:8501")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
