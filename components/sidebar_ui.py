"""
sidebar_ui.py
=============
Renders the always-visible sidebar containing:
  - App logo (small)
  - System status section with online/offline indicators
  - Recent searches (last 5) as clickable pills
  - Quick access: 10 common medicine buttons
  - How to use: 3-step mini guide
  - Keyboard shortcuts

Author:  ANTIGRAVITY BUILD
Version: 1.0.0
Date:    2026-04-18
"""

import streamlit as st
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# COMMON MEDICINE QUICK ACCESS
# ─────────────────────────────────────────────────────────────────────

QUICK_ACCESS_MEDICINES = [
    "paracetamol", "ibuprofen", "amoxicillin",
    "metformin", "cetirizine", "omeprazole",
    "aspirin", "atorvastatin", "salbutamol", "amlodipine"
]


def render_sidebar(df: pd.DataFrame) -> None:
    """
    Render the complete sidebar content.

    Sections:
      1. Logo (small version)
      2. SYSTEM STATUS: offline mode, models, DB count, no API
      3. RECENT SEARCHES: last 5 searched names as clickable pills
      4. QUICK ACCESS: buttons for 10 common medicines
      5. HOW TO USE: 3-step guide
      6. Keyboard shortcuts note

    Args:
        df (pd.DataFrame): Loaded medicine DataFrame (for medicine count).
    """
    with st.sidebar:
        # ── Small Logo ─────────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-logo">'
            '<span class="logo-med">MED</span>'
            '<span class="logo-scan">SCAN</span>'
            '<span class="logo-ai"> AI</span>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.6rem;'
            'letter-spacing:0.15em;color:var(--muted);text-align:center;'
            'margin-bottom:1.2rem;">OFFLINE MEDICINE INTELLIGENCE</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # ── System Status ──────────────────────────────────────────────
        st.markdown(
            '<div class="info-card-title" style="padding:0.3rem 0 0.6rem 0;">⚙️ SYSTEM STATUS</div>',
            unsafe_allow_html=True
        )

        med_count = len(df) if df is not None else 0
        models_loaded = st.session_state.get("models_loaded", False)

        status_items = [
            ("✓ Offline Mode Active", "var(--ok)"),
            (f"{'✓' if models_loaded else '⋯'} ML Models {'Ready' if models_loaded else 'Loading...'}", "var(--ok)"),
            (f"✓ Database: {med_count} medicines", "var(--ok)"),
            ("✓ No API Key Required", "var(--ok)"),
            ("✓ Zero Internet Calls", "var(--ok)"),
        ]

        for text, color in status_items:
            st.markdown(
                f'<div class="status-item">'
                f'<div class="status-dot"></div>'
                f'<span style="color:{color};">{text}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # ── Recent Searches ────────────────────────────────────────────
        st.markdown(
            '<div class="info-card-title" style="padding:0.3rem 0 0.6rem 0;">🕐 RECENT SEARCHES</div>',
            unsafe_allow_html=True
        )

        history = st.session_state.get("search_history", [])

        if not history:
            st.markdown(
                '<span style="font-family:var(--font-mono);font-size:0.72rem;'
                'color:var(--muted);">No searches yet.</span>',
                unsafe_allow_html=True
            )
        else:
            # Show last 5 unique searches
            recent_unique = []
            seen = set()
            for item in reversed(history):
                if item.lower() not in seen:
                    seen.add(item.lower())
                    recent_unique.append(item)
                if len(recent_unique) >= 5:
                    break

            for term in recent_unique:
                if st.button(
                    f"🔍 {term.title()}",
                    key=f"recent_{term}",
                    use_container_width=True,
                    help=f"Search for {term} again"
                ):
                    # Set the search term and switch to scan tab
                    st.session_state.sidebar_search_trigger = term
                    st.rerun()

        st.divider()

        # ── Quick Access ───────────────────────────────────────────────
        st.markdown(
            '<div class="info-card-title" style="padding:0.3rem 0 0.6rem 0;">⚡ QUICK ACCESS</div>',
            unsafe_allow_html=True
        )

        for med_name in QUICK_ACCESS_MEDICINES:
            if st.button(
                med_name.title(),
                key=f"quick_{med_name}",
                use_container_width=True,
                help=f"View {med_name} information"
            ):
                st.session_state.sidebar_search_trigger = med_name
                st.rerun()

        st.divider()

        # ── How to Use ─────────────────────────────────────────────────
        st.markdown(
            '<div class="info-card-title" style="padding:0.3rem 0 0.6rem 0;">📖 HOW TO USE</div>',
            unsafe_allow_html=True
        )

        steps = [
            ("1️⃣", "Search or Scan", "Type a medicine name or upload a medicine package photo"),
            ("2️⃣", "View Information", "Read complete offline medicine data across 9 sections"),
            ("3️⃣", "Chat with MedBot", "Ask follow-up questions about the medicine"),
        ]

        for icon, title, desc in steps:
            st.markdown(
                f'<div style="background:var(--bg-card2);border:1px solid var(--border2);'
                f'border-radius:10px;padding:0.7rem 0.8rem;margin-bottom:0.5rem;">'
                f'<div style="font-family:var(--font-mono);font-size:0.72rem;'
                f'color:var(--accent);margin-bottom:0.2rem;">{icon} {title}</div>'
                f'<div style="font-family:var(--font-ui);font-size:0.78rem;color:var(--text-sec);">'
                f'{desc}</div></div>',
                unsafe_allow_html=True
            )

        st.divider()

        # ── Version Info ───────────────────────────────────────────────
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.6rem;'
            'color:var(--muted);text-align:center;">'
            'MEDSCAN AI v1.0.0 · ANTIGRAVITY BUILD<br>'
            '100% Offline · No API Key · Python + Streamlit'
            '</div>',
            unsafe_allow_html=True
        )
