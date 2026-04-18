"""
compare_ui.py
=============
This component renders the medicine comparison tab (Tab 3).

Workflow:
  1. User selects 2 or 3 medicines from a multiselect dropdown
  2. Click 'Compare Medicines' button
  3. A side-by-side comparison table is shown (st.dataframe)
  4. A radar chart is rendered comparing scores across 5 dimensions
  5. A plain-English verdict explains which medicine is better and why

The comparison logic (scoring + chart + verdict) lives in
modules/compare_engine.py. This component is only responsible
for the Streamlit UI structure around it.

Session state:
  compare_medicines list is stored in session state so it persists
  across reruns and can be pre-populated from the database browser (Tab 2)
  when a user clicks "Add to comparison" on a medicine card.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import streamlit as st
import pandas as pd
from modules.compare_engine import (
    compare_medicines,
    generate_radar_chart,
    generate_comparison_verdict,
    build_comparison_table,
)
from modules.medicine_db import get_medicine_by_key


def render_compare_tab(df: pd.DataFrame) -> None:
    """
    Render the full medicine comparison tab interface.

    Sections:
      1. Section header
      2. Medicine multi-select (up to 3)
      3. [COMPARE] button
      4. Comparison table (if medicines selected)
      5. Radar chart
      6. Comparison verdict
      7. Download comparison CSV

    Args:
        df (pd.DataFrame): Loaded medicine DataFrame.
    """
    st.markdown(
        '<div class="section-header">■ COMPARE MEDICINES — SIDE BY SIDE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    # Get sorted list of all medicine names for the multiselect
    all_names = sorted(df["name"].tolist())

    # Pre-populate from session state if medicines were added from browser
    default_selection = st.session_state.get("compare_medicines", [])
    # Filter to valid names
    default_selection = [n for n in default_selection if n in all_names]

    # ── Medicine Selection ────────────────────────────────────────────
    selected_names = st.multiselect(
        label="Select medicines to compare (2–3):",
        options=all_names,
        default=default_selection[:3],
        max_selections=3,
        key="compare_multiselect",
        help="Select 2 or 3 medicines to compare side by side."
    )

    # Update session state
    st.session_state.compare_medicines = selected_names

    col_btn, col_reset = st.columns([2, 1])
    with col_btn:
        compare_clicked = st.button("⚖️ Compare Medicines", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.compare_medicines = []
            st.rerun()

    # ── Validation ────────────────────────────────────────────────────
    if compare_clicked or st.session_state.get("compare_mode", False):
        if len(selected_names) < 2:
            st.markdown(
                '<div class="info-card" style="border-color:rgba(245,158,11,0.3);border-radius:12px;">'
                '<span style="color:var(--warn);font-family:var(--font-mono);font-size:0.85rem;">'
                '⚠️ Please select at least 2 medicines to compare.'
                '</span></div>',
                unsafe_allow_html=True
            )
            return

        if len(selected_names) > 3:
            st.markdown(
                '<div class="info-card" style="border-color:rgba(239,68,68,0.3);">'
                '<span style="color:var(--danger);font-family:var(--font-mono);font-size:0.85rem;">'
                '❌ Maximum 3 medicines can be compared at once.'
                '</span></div>',
                unsafe_allow_html=True
            )
            return

        # ── Load medicine data ─────────────────────────────────────────
        medicines = []
        for name in selected_names:
            med = get_medicine_by_key(df, name)
            if med:
                medicines.append(med)

        if len(medicines) < 2:
            st.error("Could not load medicine data. Please try again.")
            return

        st.session_state.compare_mode = True

        # ── Comparison Table ───────────────────────────────────────────
        st.markdown("---")
        render_comparison_table(medicines)

        # ── Radar Chart ────────────────────────────────────────────────
        render_radar_chart(medicines)

        # ── Verdict ───────────────────────────────────────────────────
        render_comparison_verdict(medicines)


def render_comparison_table(medicines: list) -> None:
    """
    Render side-by-side comparison table using st.dataframe().

    Args:
        medicines (list[dict]): List of 2 or 3 medicine dicts.
    """
    st.markdown(
        '<div class="section-header">■ SIDE-BY-SIDE COMPARISON TABLE</div>',
        unsafe_allow_html=True
    )

    try:
        comp_df = build_comparison_table(medicines)
        st.dataframe(
            comp_df,
            use_container_width=True,
            height=400,
        )
    except Exception as e:
        st.error(f"Error building comparison table: {e}")


def render_radar_chart(medicines: list) -> None:
    """
    Render the matplotlib radar chart via st.pyplot().

    Args:
        medicines (list[dict]): List of medicine dicts.
    """
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem;">■ RADAR CHART COMPARISON</div>',
        unsafe_allow_html=True
    )

    try:
        from modules.compare_engine import compute_medicine_scores
        scores = [compute_medicine_scores(m) for m in medicines]
        fig = generate_radar_chart(medicines, scores)
        st.pyplot(fig, use_container_width=False)
    except Exception as e:
        st.error(f"Could not render radar chart: {e}")


def render_comparison_verdict(medicines: list) -> None:
    """
    Render the AI-generated comparison verdict.

    Args:
        medicines (list[dict]): List of medicine dicts.
    """
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem;">■ COMPARISON VERDICT</div>',
        unsafe_allow_html=True
    )

    try:
        from modules.compare_engine import compute_medicine_scores
        scores = [compute_medicine_scores(m) for m in medicines]
        verdict = generate_comparison_verdict(medicines, scores)

        st.markdown(
            f'<div class="compare-card">{verdict}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Could not generate verdict: {e}")
