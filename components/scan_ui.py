"""
scan_ui.py
==========
This component renders the main medicine scan and search interface (Tab 1).

Two input modes:
  1. Upload Image mode:
     - User uploads a JPG/PNG/WEBP photo of a medicine package
     - Image is previewed and passed through the OCR pipeline
     - Extracted text candidates are shown in an expandable panel
     - The best candidate is automatically searched in the database

  2. Type Name mode:
     - User types a medicine name, brand name, or generic name
     - The search engine tries exact → TF-IDF → fuzzy strategies
     - Results appear immediately with a confidence badge

After a medicine is found (by either mode), this component:
  - Calls render_medicine_header() from medicine_card.py
  - Calls render_all_sections() from info_sections.py
  - Calls render_chatbot() from chatbot_ui.py

If no medicine is found, a 'not found' card is shown with suggestions.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io


def render_scan_hero() -> None:
    """
    Render the minimalist hero text shown when no medicine is loaded.
    Matches MedScan+ screenshot aesthetic.
    """
    st.markdown(
        '<div class="scan-hero" style="text-align: center; margin-top: 2rem;">'
        '<h1 style="font-family: var(--font-display); font-size: 3rem; font-weight: 800; color: #fff; margin-bottom:0.2rem;">MedScan+</h1>'
        '<p style="font-family: var(--font-ui); font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem;">Identify medicines instantly</p>'
        '</div>',
        unsafe_allow_html=True
    )


def render_scan_interface(df: pd.DataFrame, vectorizer, matrix) -> None:
    from modules.medicine_search import search_medicine
    from modules.ocr_engine import run_ocr_pipeline
    from components.medicine_card import render_medicine_header, render_search_strategy_badge
    from components.info_sections import render_all_sections
    from components.chatbot_ui import render_chatbot

    if not st.session_state.get("current_medicine"):
        render_scan_hero()

    # ── Scan Mode Selector ────────────────────────────────────────────
    st.markdown('<div class="section-header" style="text-align:center; border:none; color: var(--muted);">CHOOSE INPUT METHOD</div>', unsafe_allow_html=True)

    scan_mode = st.radio(
        label="Choose input method:",
        options=["📷 Camera", "📁 Upload", "⌨️ Type"],
        horizontal=True,
        key="scan_mode_radio",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # MODE A/B: CAMERA OR IMAGE UPLOAD + OCR
    # ══════════════════════════════════════════════════════════════════
    if "Camera" in scan_mode or "Upload" in scan_mode:
        uploaded_file = None
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if "Camera" in scan_mode:
                uploaded_file = st.camera_input("Point camera at a medicine package", label_visibility="collapsed")
            else:
                uploaded_file = st.file_uploader(
                    label="Upload medicine image",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="image_uploader",
                    label_visibility="collapsed"
                )

        if uploaded_file:
            col_img, col_info = st.columns([1, 2])
            with col_img:
                try:
                    preview_img = Image.open(io.BytesIO(uploaded_file.getvalue()))
                    st.image(preview_img, use_container_width=True)
                except Exception:
                    st.warning("Could not preview image.")

            with col_info:
                scan_btn = st.button("SCAN", use_container_width=True, key="scan_btn")

            if scan_btn:
                with st.spinner("Building monograph..."):
                    uploaded_file.seek(0)
                    ocr_result = run_ocr_pipeline(uploaded_file)
                    st.session_state.ocr_result = ocr_result

            if st.session_state.get("ocr_result"):
                render_ocr_results(st.session_state.ocr_result, df, vectorizer, matrix)
        else:
            st.markdown(
                '<div style="text-align:center; font-family:var(--font-ui); font-size:0.9rem; color:var(--muted); margin-top: 1rem;">'
                'Point your camera at a medicine package to get<br>detailed information about the medication'
                '</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════════
    # MODE B: TEXT SEARCH
    # ══════════════════════════════════════════════════════════════════
    else:
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.72rem;'
            'color:var(--muted);margin-bottom:0.5rem;">'
            'Type a medicine name, brand name, or generic name to search.'
            '</div>',
            unsafe_allow_html=True
        )

        # Check if sidebar quick-access or recent search was triggered
        sidebar_trigger = st.session_state.pop("sidebar_search_trigger", None)

        col_search, col_btn = st.columns([4, 1])
        with col_search:
            search_query = st.text_input(
                label="Search Medicine",
                placeholder="e.g. Paracetamol, Eyemist, Zyrtec, Crocin...",
                value=sidebar_trigger or "",
                key="medicine_search_input",
                label_visibility="collapsed"
            )
        with col_btn:
            search_clicked = st.button("🔍 Search", use_container_width=True, key="search_btn")

        # Auto-trigger if sidebar link was clicked
        if sidebar_trigger:
            search_clicked = True

        # ── Perform Search ─────────────────────────────────────────────
        if search_clicked and search_query:
            with st.spinner(f"Searching for '{search_query}'..."):
                result = search_medicine(
                    query=search_query,
                    df=df,
                    vectorizer=vectorizer,
                    matrix=matrix
                )

            if result["found"]:
                # Store result in session state + update search history
                st.session_state.current_medicine = result["medicine"]
                st.session_state.chat_history = []  # reset chat for new medicine

                # Update search history (deduplicated, latest first)
                med_name = result["medicine"].get("name", search_query)
                history = st.session_state.get("search_history", [])
                if med_name not in history:
                    history.insert(0, med_name)
                    st.session_state.search_history = history[:10]

            else:
                # Not found — show suggestions
                st.session_state.current_medicine = None
                render_not_found(search_query, result.get("suggestions", []))

        # Show medicine info if one is loaded
        if not search_clicked and not st.session_state.get("current_medicine"):
            render_scan_hero()

    # ══════════════════════════════════════════════════════════════════
    # MEDICINE INFO DISPLAY (shared for both modes)
    # ══════════════════════════════════════════════════════════════════
    if st.session_state.get("current_medicine"):
        medicine = st.session_state.current_medicine
        strategy = "exact"
        confidence = 1.0
        search_time = 0.0

        # If we have a recent search result, use its metadata
        # (not stored in session to keep state clean — use defaults)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Render strategy badge
        render_search_strategy_badge(strategy, confidence, search_time)

        # Render main medicine info components
        render_medicine_header(medicine)
        render_all_sections(medicine)

        # Render chatbot using pre-built intent index from session state
        chat_vec = st.session_state.get("tfidf_vec")
        chat_mat = st.session_state.get("tfidf_mat")
        chat_keys = st.session_state.get("tfidf_keys")

        if chat_vec and chat_mat is not None and chat_keys:
            render_chatbot(medicine, chat_vec, chat_mat, chat_keys)
        else:
            st.info("🤖 MedBot chatbot is loading... Please wait a moment and try again.")


def render_ocr_results(ocr_result: dict, df, vectorizer, matrix) -> None:
    """
    Display the OCR pipeline results in an expandable section.

    Shows: preprocessed image, extracted text, candidate pills.
    If candidates found, automatically runs search for best candidate.

    Args:
        ocr_result (dict): Result dict from run_ocr_pipeline().
        df: Medicine DataFrame.
        vectorizer: Search vectoriser.
        matrix: Search matrix.
    """
    from modules.medicine_search import search_medicine

    if not ocr_result.get("success"):
        error_msg = ocr_result.get("error", "Unknown OCR error")

        # Check if Tesseract-specific error
        if "tesseract" in error_msg.lower() or "not found" in error_msg.lower():
            st.markdown(
                '<div class="not-found-box" style="border-color:rgba(245,158,11,0.3);">'
                '<div class="not-found-title" style="color:var(--warn);">⚠️ Tesseract Not Installed</div>'
                '<div style="font-family:var(--font-ui);font-size:0.85rem;color:var(--text-sec);margin-top:0.5rem;">'
                'OCR requires Tesseract installed on your system:<br><br>'
                '• <strong>Mac:</strong> <code>brew install tesseract</code><br>'
                '• <strong>Linux:</strong> <code>sudo apt install tesseract-ocr</code><br>'
                '• <strong>Windows:</strong> <a href="https://github.com/UB-Mannheim/tesseract/wiki" target="_blank">Download from GitHub</a><br><br>'
                'You can still use the <strong>Type Name</strong> mode to search without Tesseract.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(f"❌ OCR failed: {error_msg}")

        st.markdown(
            '<div class="info-card" style="margin-top:1rem;">'
            '<span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--muted);">'
            'Tip: Use the ⌨️ Type Name mode to search by typing the medicine name directly.'
            '</span></div>',
            unsafe_allow_html=True
        )
        return

    # ── Show OCR output in expandable section ──────────────────────────
    with st.expander("🔍 OCR Pipeline Results", expanded=True):
        col_img, col_text = st.columns(2)

        with col_img:
            st.markdown(
                '<div class="info-card-title">📸 PREPROCESSED IMAGE</div>',
                unsafe_allow_html=True
            )
            preprocessed = ocr_result.get("preprocessed_image")
            if preprocessed is not None:
                # Convert numpy array to PIL for display
                display_img = Image.fromarray(preprocessed.astype("uint8"))
                st.image(display_img, caption="After OCR preprocessing", use_container_width=True)

        with col_text:
            st.markdown(
                '<div class="info-card-title">📝 EXTRACTED TEXT</div>',
                unsafe_allow_html=True
            )
            cleaned = ocr_result.get("cleaned_text", "")
            if cleaned:
                st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""),
                        language=None)
            else:
                st.markdown(
                    '<span style="color:var(--muted);font-size:0.85rem;">'
                    'Could not extract text. Try a clearer image.</span>',
                    unsafe_allow_html=True
                )

        # ── Candidate Pills ────────────────────────────────────────────
        candidates = ocr_result.get("candidates", [])
        if candidates:
            st.markdown(
                '<div class="info-card-title" style="margin-top:0.8rem;">🎯 DETECTED NAME CANDIDATES</div>',
                unsafe_allow_html=True
            )
            pills_html = ""
            for i, cand in enumerate(candidates):
                is_best = " best" if i == 0 else ""
                pills_html += f'<span class="ocr-candidate-pill{is_best}">{cand}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)

            # Auto-search for best candidate
            best_candidate = ocr_result.get("best_candidate", "")
            if best_candidate:
                st.markdown(
                    f'<div style="font-family:var(--font-mono);font-size:0.72rem;'
                    f'color:var(--muted);margin-top:0.5rem;">🔍 Searching for: '
                    f'<strong style="color:var(--accent);">{best_candidate}</strong></div>',
                    unsafe_allow_html=True
                )

                result = search_medicine(
                    query=best_candidate, df=df,
                    vectorizer=vectorizer, matrix=matrix
                )

                if result["found"]:
                    st.session_state.current_medicine = result["medicine"]
                    st.session_state.chat_history = []
                    st.success(f"✅ Medicine identified: **{result['medicine']['name'].title()}**")
                else:
                    # Try other candidates
                    for cand in candidates[1:]:
                        result = search_medicine(cand, df, vectorizer, matrix)
                        if result["found"]:
                            st.session_state.current_medicine = result["medicine"]
                            st.session_state.chat_history = []
                            st.success(f"✅ Found via candidate '{cand}': **{result['medicine']['name'].title()}**")
                            break
                    else:
                        render_not_found(best_candidate, result.get("suggestions", []))
        else:
            st.markdown(
                '<div class="not-found-box">'
                '<div class="not-found-title">💬 No Text Detected</div>'
                '<div style="font-family:var(--font-ui);font-size:0.85rem;color:var(--text-sec);">'
                'Could not extract recognisable text from the image.<br>'
                'Tips: Use better lighting, ensure text is sharp, or try the Type Name mode.'
                '</div></div>',
                unsafe_allow_html=True
            )


def render_not_found(query: str, suggestions: list) -> None:
    """
    Render the 'medicine not found' state with suggestion pills.

    Args:
        query (str): The search query that returned no results.
        suggestions (list[str]): List of suggested medicine names.
    """
    st.markdown(
        f'<div class="not-found-box">'
        f'<div class="not-found-title">❌ Medicine Not Found</div>'
        f'<div style="font-family:var(--font-ui);font-size:0.88rem;color:var(--text-sec);margin-top:0.5rem;">'
        f'No results for <strong>"{query}"</strong> in the offline database.'
        f'</div>',
        unsafe_allow_html=True
    )

    if suggestions:
        st.markdown(
            '<div style="margin-top:0.8rem;font-family:var(--font-mono);'
            'font-size:0.72rem;color:var(--muted);">Did you mean:</div>',
            unsafe_allow_html=True
        )
        pills_html = "".join(
            f'<span class="ocr-candidate-pill">{s.title()}</span>'
            for s in suggestions
        )
        st.markdown(pills_html + '</div>', unsafe_allow_html=True)

        # Clickable suggestion buttons
        if suggestions:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            cols = st.columns(len(suggestions))
            for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(
                        f"→ {suggestion.title()}",
                        key=f"sugg_{i}_{suggestion}",
                        use_container_width=True
                    ):
                        st.session_state.sidebar_search_trigger = suggestion
                        st.rerun()
    else:
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.72rem;'
            'color:var(--muted);margin-top:0.5rem;">'
            'This medicine is not in the current offline database. '
            'See docs/DATASET_GUIDE.md to expand the database.'
            '</div>',
            unsafe_allow_html=True
        )
