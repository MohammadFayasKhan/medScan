"""
app.py
======
Main entry point for MedScan AI. Run this file to launch the app:

    streamlit run app.py

What this file does:
  1. Configures the Streamlit page (title, icon, layout)
  2. Injects the custom CSS design system
  3. Initialises session state variables on first run
  4. Loads all ML models (cached so they only load once per session)
  5. Renders the hero header with the app title
  6. Renders the sidebar (system status, recent searches, quick access)
  7. Sets up the 3 main navigation tabs:
       Tab 1: Scan and Search (OCR upload + text search + chatbot)
       Tab 2: Medicine Database (filterable grid of all 20 medicines)
       Tab 3: Compare Medicines (side-by-side radar chart comparison)

Model loading strategy:
  @st.cache_resource ensures that all ML models (search index, Naive Bayes
  classifier, TF-IDF vectoriser) are loaded only once and shared across
  all users in the same Streamlit session. On reruns (which happen on every
  interaction), the cached result is returned instantly.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import os
import sys
import logging

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# PATH SETUP
# Ensure project root is on sys.path so all imports resolve correctly
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────
# IMPORTS — MODULES
# ─────────────────────────────────────────────────────────────────────
from modules.preprocessor import download_nltk_resources
from modules.medicine_db import load_database, get_all_names
from modules.medicine_search import build_search_index
from modules.medicine_db import build_search_corpus
from modules.chatbot import load_intents, build_intent_index
from modules.intent_classifier import init_classifier
from modules.model_trainer import train_and_save_search_index, load_search_index

# ─────────────────────────────────────────────────────────────────────
# IMPORTS — COMPONENTS
# ─────────────────────────────────────────────────────────────────────
from components.ui_styles import inject_styles
from components.sidebar_ui import render_sidebar
from components.scan_ui import render_scan_interface
from components.compare_ui import render_compare_tab

# ─────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: PAGE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedScan AI — Offline Medicine Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "MEDSCAN AI v1.0.0 — MedScan AI Project — Offline Medicine Intelligence System",
    }
)

# ═════════════════════════════════════════════════════════════════════
# SECTION 2: INJECT CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════

inject_styles()

# ═════════════════════════════════════════════════════════════════════
# SECTION 3: SESSION STATE INITIALISATION
# Must be done before any other Streamlit rendering to avoid AttributeError
# ═════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """
    Initialise all session state variables with safe defaults.
    Only sets values if they don't already exist (first run).
    """
    defaults = {
        "current_medicine":    None,    # currently selected medicine dict
        "chat_history":        [],      # list of {role, text, intent} dicts
        "search_history":      [],      # last 10 searched name strings
        "ocr_result":          None,    # last OCR pipeline output dict
        "scan_mode":           "text",  # "upload" | "text"
        "compare_medicines":   [],      # up to 3 medicine names for comparison
        "compare_mode":        False,   # True when comparison is showing
        "tfidf_vec":           None,    # fitted TfidfVectorizer (chatbot intent)
        "tfidf_mat":           None,    # TF-IDF pattern matrix (chatbot intent)
        "tfidf_keys":          None,    # intent labels list (chatbot)
        "nb_classifier":       None,    # trained Naive Bayes classifier
        "label_encoder":       None,    # LabelEncoder for NB classifier
        "medicine_df":         None,    # full medicine DataFrame
        "search_vec":          None,    # search TF-IDF vectoriser
        "search_mat":          None,    # search TF-IDF matrix
        "models_loaded":       False,   # True after all models initialised
        "active_tab":          "scan",  # "scan" | "database" | "compare"
        "show_toast":          None,    # {message, type} for notifications
        "zoom_level":          1,       # reserved for future image zoom
        "dark_mode":           True,    # always True (dark-only app)
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


init_session_state()


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL LOADING
# Load all ML models on first run; use already-loaded models on reruns
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Load or train all ML models. Cached as a resource (shared across sessions).

    Returns:
        dict: All model components needed by the app.
    """
    # Step 1: Ensure NLTK resources are available
    download_nltk_resources()

    # Step 2: Load medicine database
    df = load_database()

    # Step 3: Build/load search index
    saved_search = load_search_index()
    if saved_search is not None:
        search_vec, search_mat = saved_search
    else:
        corpus = build_search_corpus(df)
        search_vec, search_mat = build_search_index(corpus)
        # Save for future sessions
        try:
            from modules.model_trainer import train_and_save_search_index
            train_and_save_search_index(df)
        except Exception:
            pass  # Non-fatal; will work without saved index

    # Step 4: Train/load Naive Bayes intent classifier
    nb_vec, nb_clf, nb_enc = init_classifier()

    # Step 5: Build chatbot TF-IDF intent index
    intents = load_intents()
    chat_vec, chat_mat, chat_labels = build_intent_index(intents)

    return {
        "df": df,
        "search_vec": search_vec,
        "search_mat": search_mat,
        "nb_vec": nb_vec,
        "nb_clf": nb_clf,
        "nb_enc": nb_enc,
        "chat_vec": chat_vec,
        "chat_mat": chat_mat,
        "chat_labels": chat_labels,
    }


# Load models with loading indicator
if not st.session_state.models_loaded:
    with st.spinner("⚙️ Initialising MEDSCAN AI — Loading offline models..."):
        try:
            models = load_all_models()

            # Populate session state with loaded models
            st.session_state.medicine_df   = models["df"]
            st.session_state.search_vec    = models["search_vec"]
            st.session_state.search_mat    = models["search_mat"]
            st.session_state.nb_classifier = models["nb_clf"]
            st.session_state.label_encoder = models["nb_enc"]
            st.session_state.tfidf_vec     = models["chat_vec"]
            st.session_state.tfidf_mat     = models["chat_mat"]
            st.session_state.tfidf_keys    = models["chat_labels"]
            st.session_state.models_loaded = True

        except FileNotFoundError as e:
            st.error(
                f"❌ **Setup Error:** {e}\n\n"
                "Please run `python setup.py` first to set up the database and models."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ **Unexpected error during model loading:** {e}")
            logger.exception("Model loading failed")
            st.stop()

# Convenience shorthand references to session state models
df          = st.session_state.medicine_df
search_vec  = st.session_state.search_vec
search_mat  = st.session_state.search_mat
chat_vec    = st.session_state.tfidf_vec
chat_mat    = st.session_state.tfidf_mat
chat_labels = st.session_state.tfidf_keys


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: HERO HEADER
# ═════════════════════════════════════════════════════════════════════

# Header HTML with split-color wordmark and MedScan AI Project badge
hero_html = """
<div class="hero">
  <div style="display:flex;align-items:center;justify-content:center;gap:1.5rem;flex-wrap:wrap;">
    <div>
      <div class="logo-wrap">
        <span class="logo-med">MED</span><span class="logo-scan">SCAN</span><span class="logo-ai">&nbsp;AI</span>
      </div>
      <div class="hero-subtitle">Offline Medicine Intelligence System</div>
    </div>
    <div class="antigravity-badge">
      <span class="badge-ag">INT428</span>&nbsp;<span class="badge-build">AI SYSTEMS</span>
    </div>
  </div>
  <div style="margin-top:0.8rem;font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);">
    💊 20 Medicines &nbsp;·&nbsp; 🤖 NLP Chatbot &nbsp;·&nbsp; 📷 OCR Scanner &nbsp;·&nbsp; 🔒 100% Offline
  </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: SIDEBAR
# ═════════════════════════════════════════════════════════════════════

render_sidebar(df)


# ═════════════════════════════════════════════════════════════════════
# SECTION 7: NAVIGATION TABS
# ═════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "💊 Scan & Search",
    "📋 Medicine Database",
    "⚖️ Compare Medicines",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1: SCAN & SEARCH
# ─────────────────────────────────────────────────────────────────────
with tab1:
    render_scan_interface(
        df=df,
        vectorizer=search_vec,
        matrix=search_mat
    )


# ─────────────────────────────────────────────────────────────────────
# TAB 2: MEDICINE DATABASE BROWSER
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-header">■ OFFLINE MEDICINE DATABASE BROWSER</div>',
        unsafe_allow_html=True
    )

    # ── Filter Bar ─────────────────────────────────────────────────────
    col_search, col_cat = st.columns([3, 2])

    with col_search:
        db_filter = st.text_input(
            label="Filter by name",
            placeholder="Filter medicines...",
            key="db_filter_input",
            label_visibility="collapsed"
        )

    with col_cat:
        from modules.medicine_db import get_all_categories, filter_by_category
        all_cats = ["All"] + get_all_categories(df)
        selected_cat = st.selectbox(
            label="Category",
            options=all_cats,
            key="db_cat_filter",
            label_visibility="collapsed"
        )

    # Apply filters
    filtered_df = filter_by_category(df, selected_cat) if selected_cat != "All" else df
    if db_filter:
        q = db_filter.lower()
        filtered_df = filtered_df[
            filtered_df["name"].str.lower().str.contains(q, na=False) |
            filtered_df["generic_name"].str.lower().str.contains(q, na=False) |
            filtered_df["category"].str.lower().str.contains(q, na=False)
        ]

    # ── Medicine Count ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--muted);'
        f'margin-bottom:0.8rem;">Showing <strong style="color:var(--accent);">'
        f'{len(filtered_df)}</strong> of {len(df)} medicines in offline database</div>',
        unsafe_allow_html=True
    )

    # ── 3-Column Medicine Card Grid ─────────────────────────────────────
    from modules.medicine_db import get_medicine_card_data

    if filtered_df.empty:
        st.markdown(
            '<div class="not-found-box"><div class="not-found-title">No medicines match your filter.</div></div>',
            unsafe_allow_html=True
        )
    else:
        # Display in 3 columns
        cols = st.columns(3)

        for i, (_, row) in enumerate(filtered_df.iterrows()):
            card_data = get_medicine_card_data(row)
            col = cols[i % 3]

            with col:
                # Render card HTML
                card_html = f"""
                <div class="db-card">
                  <div class="db-card-accent" style="background:{card_data['color_key']};"></div>
                  <div class="db-card-name">{card_data['name'].title()}</div>
                  <div class="db-card-detail">{card_data['generic_name'].split('/')[0].strip()}</div>
                  <div class="db-card-detail" style="margin-top:0.3rem;">{card_data['form'].split('/')[0].strip()} · {card_data['strength'].split('/')[0].strip()}</div>
                  <div style="margin-top:0.5rem;">
                    <span class="pack-tag">{card_data['category'].split('/')[0].strip()}</span>
                  </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # Open button — loads medicine in Tab 1
                if st.button(
                    f"View {card_data['name'].title()}",
                    key=f"db_card_{i}_{card_data['name']}",
                    use_container_width=True
                ):
                    from modules.medicine_db import get_medicine_by_key
                    medicine = get_medicine_by_key(df, card_data["name"])
                    if medicine:
                        st.session_state.current_medicine = medicine
                        st.session_state.chat_history = []
                        st.session_state.active_tab = "scan"
                        # Update search history
                        hist = st.session_state.search_history
                        if card_data["name"] not in hist:
                            hist.insert(0, card_data["name"])
                            st.session_state.search_history = hist[:10]
                        st.rerun()

    # ── Export All Button ───────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    csv_data = df.to_csv(index=False)
    from modules.export_utils import create_download_button
    create_download_button(
        data=csv_data,
        filename="medscan_database_export.csv",
        mime="text/csv",
        label="⬇ Export Full Database (CSV)"
    )


# ─────────────────────────────────────────────────────────────────────
# TAB 3: COMPARE MEDICINES
# ─────────────────────────────────────────────────────────────────────
with tab3:
    render_compare_tab(df=df)
