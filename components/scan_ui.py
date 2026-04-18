"""
scan_ui.py
==========
Renders the medicine scanning and text search interface (Tab 1).

Sections:
  - Scan mode selector: [📷 Upload Image] or [⌨️ Type Name]
  - Upload mode: file uploader + preview + SCAN button → OCR pipeline
  - Text mode: text input + SEARCH button + autocomplete
  - Not-found state with suggestions
  - OCR results display (in expandable section)

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io


def render_scan_hero() -> None:
    """
    Render the pulsing scan circle hero section shown when no medicine is loaded.
    Displayed before any search or scan is performed.
    """
    st.markdown(
        '<div class="scan-hero">'
        '<div class="scan-circle">💊</div>'
        '<div class="scan-title">Scan or Search a Medicine</div>'
        '<div class="scan-subtitle">'
        'UPLOAD AN IMAGE · TYPE A NAME · INSTANT OFFLINE IDENTIFICATION'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


def render_scan_interface(df: pd.DataFrame, vectorizer, matrix) -> None:
    """
    Render the full scan/search interface and handle all user interactions.

    Manages:
      - Scan mode radio selection
      - File uploader + OCR pipeline for image mode
      - Text input + multi-strategy search for text mode
      - Results display (medicine card + sections + chatbot)
      - Not-found state with suggestions

    Args:
        df (pd.DataFrame): Loaded medicine DataFrame.
        vectorizer: Fitted TF-IDF search vectoriser.
        matrix: TF-IDF search matrix.
    """
    from modules.medicine_search import search_medicine
    from modules.ocr_engine import run_ocr_pipeline
    from components.medicine_card import render_medicine_header, render_search_strategy_badge
    from components.info_sections import render_all_sections
    from components.chatbot_ui import render_chatbot

    # ── Scan Mode Selector ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">■ SCAN MODE</div>',
        unsafe_allow_html=True
    )

    scan_mode = st.radio(
        label="Choose input method:",
        options=["📷 Upload Image", "⌨️ Type Name"],
        horizontal=True,
        key="scan_mode_radio",
        label_visibility="collapsed"
    )
    st.session_state.scan_mode = "upload" if "Upload" in scan_mode else "text"

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # MODE A: IMAGE UPLOAD + OCR
    # ══════════════════════════════════════════════════════════════════
    if st.session_state.scan_mode == "upload":
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.72rem;'
            'color:var(--muted);margin-bottom:0.5rem;">'
            'Upload a clear photo of a medicine package, box, or strip label.'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            label="Upload medicine image",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_uploader",
            label_visibility="collapsed",
            help="Supported: JPG, PNG, WEBP. Ensure text is clearly visible."
        )

        if uploaded_file:
            # ── File size check (warn if > 10MB) ─────────────────────
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 10:
                st.warning(
                    f"⚠️ Large image ({file_size_mb:.1f}MB). "
                    "Processing may be slow. Consider using a smaller image for faster OCR."
                )

            # ── Show small image preview ───────────────────────────────
            col_img, col_info = st.columns([1, 2])
            with col_img:
                # Reset file position after size check read
                uploaded_file.seek(0)
                try:
                    preview_img = Image.open(io.BytesIO(uploaded_file.read()))
                    st.image(preview_img, caption="Uploaded Image", use_container_width=True)
                    uploaded_file.seek(0)
                except Exception:
                    st.warning("Could not preview image.")
                    uploaded_file.seek(0)

            with col_info:
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="info-card-title">📋 FILE INFO</div>'
                    f'<div class="info-row"><span class="info-label">Name</span>'
                    f'<span class="info-value">{uploaded_file.name}</span></div>'
                    f'<div class="info-row"><span class="info-label">Size</span>'
                    f'<span class="info-value">{file_size_mb:.2f} MB</span></div>'
                    f'<div class="info-row"><span class="info-label">Type</span>'
                    f'<span class="info-value">{uploaded_file.type}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                scan_btn = st.button("▶ SCAN IMAGE", use_container_width=True, key="scan_btn")

            # ── Run OCR on button click ────────────────────────────────
            if scan_btn:
                with st.spinner("🔍 Running OCR pipeline..."):
                    uploaded_file.seek(0)
                    ocr_result = run_ocr_pipeline(uploaded_file)
                    st.session_state.ocr_result = ocr_result

            # ── Display OCR Results ────────────────────────────────────
            if st.session_state.get("ocr_result"):
                render_ocr_results(
                    st.session_state.ocr_result,
                    df, vectorizer, matrix
                )

        else:
            # No file uploaded yet — show hero
            if not st.session_state.get("current_medicine"):
                render_scan_hero()

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
