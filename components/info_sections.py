"""
info_sections.py
================
This component renders the 9 collapsible accordion sections that display
detailed medicine information below the header card.

Each section pulls specific fields from the medicine data dictionary
and renders them with appropriate styling:
  - ℹ️  Basic Information     → name, generic, form, strength, manufacturer
  - ⚡  Usage and Action      → uses, mechanism, indications
  - 💧  Dosage and Use        → dosage, timing, spacing, admin_tips
  - ⚠️  Warnings              → pregnancy, paediatric, driving, storage
  - 🚫  Contraindications     → conditions where medicine must NOT be used
  - 🔗  Drug Interactions     → known medicine-medicine interactions
  - 🔴  Side Effects          → common (amber) and serious (red) effects
  - 🛍️  Availability          → pack sizes and therapeutic substitutes
  - 📚  Sources               → reference databases used

Design decisions:
  - Warnings are shown in amber, contraindications and serious side effects
    in red, drug interactions in purple — colour coding creates visual hierarchy.
  - Streamlit expanders are used for all sections so the user can focus on
    what they care about without scrolling past irrelevant content.
  - render_all_sections() is called once from scan_ui.py after the header card.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import streamlit as st


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _info_row(label: str, value: str) -> str:
    """
    Generate an HTML info row (label + value pair) as an HTML string.

    Args:
        label (str): Field label (displayed in monospace, muted).
        value (str): Field value (displayed in normal text).

    Returns:
        str: HTML string for the row.
    """
    return f"""
    <div class="info-row">
      <span class="info-label">{label}</span>
      <span class="info-value">{value}</span>
    </div>
    """


def _bullet_list(items, dot_class: str = "") -> str:
    """
    Generate an HTML bullet list from a list or comma-separated string.

    Args:
        items: list[str] or comma-separated str.
        dot_class (str): CSS modifier class for bullet dot color.

    Returns:
        str: HTML string of bullet items.
    """
    if isinstance(items, str):
        items = [i.strip() for i in items.split(",") if i.strip()]
    if not items:
        return '<span style="color:var(--muted);font-size:0.85rem;">Not specified</span>'

    rows = []
    for item in items:
        if item and item != "Not specified":
            rows.append(
                f'<div class="bullet-item">'
                f'<div class="bullet-dot {dot_class}"></div>'
                f'<span style="font-family:var(--font-ui);font-size:0.88rem;color:var(--text);line-height:1.5;">{item}</span>'
                f'</div>'
            )
    return "".join(rows) if rows else '<span style="color:var(--muted);">Not specified</span>'


def _pack_tags(items) -> str:
    """Generate HTML pill tags for pack sizes or substitute names."""
    if isinstance(items, str):
        items = [i.strip() for i in items.split(",") if i.strip()]
    if not items:
        return '<span style="color:var(--muted);">Not specified</span>'
    return "".join(
        f'<span class="pack-tag">{item}</span>'
        for item in items if item
    )


# ─────────────────────────────────────────────────────────────────────
# SECTION RENDERERS
# ─────────────────────────────────────────────────────────────────────

def render_basic_info(medicine: dict) -> None:
    """
    Render the 'Basic Information' section in a Streamlit expander.

    Fields: generic_name, form, strength, active_substance,
            manufacturer, category, and characteristic features list.

    Args:
        medicine (dict): Full medicine data dict.
    """
    features = medicine.get("features", [])
    features_html = _bullet_list(features)

    rows_html = (
        _info_row("Generic Name", medicine.get("generic_name", "N/A"))
        + _info_row("Form", medicine.get("form", "N/A"))
        + _info_row("Strength", medicine.get("strength", "N/A"))
        + _info_row("Active Substance", medicine.get("active_substance", "N/A"))
        + _info_row("Manufacturer", medicine.get("manufacturer", "Not specified"))
        + _info_row("Category", medicine.get("category", "N/A"))
    )

    content = f"""
    <div class="info-card">
      <div class="info-card-title">ℹ️ &nbsp; BASIC INFORMATION</div>
      {rows_html}
      <div style="margin-top:1rem;">
        <div class="info-card-title" style="margin-bottom:0.5rem;">⭐ CHARACTERISTIC FEATURES</div>
        {features_html}
      </div>
    </div>
    """

    with st.expander("ℹ️  Basic Information", expanded=True):
        st.markdown(content, unsafe_allow_html=True)


def render_usage_action(medicine: dict) -> None:
    """
    Render the 'Usage and Action' section.

    Fields: mechanism, indications (list), uses.

    Args:
        medicine (dict): Full medicine data dict.
    """
    indications = medicine.get("indications", [])
    indications_html = _bullet_list(indications)

    content = f"""
    <div class="info-card">
      <div class="info-card-title">⚡ USAGE AND ACTION</div>
      <div class="info-row">
        <span class="info-label">Primary Uses</span>
        <span class="info-value">{medicine.get("uses", "Not specified")}</span>
      </div>
      <div style="margin-top:1rem;">
        <div class="info-card-title" style="margin-bottom:0.5rem;">🎯 MEDICAL INDICATIONS</div>
        {indications_html}
      </div>
      <div class="info-row" style="margin-top:1rem;">
        <span class="info-label">Mechanism</span>
        <span class="info-value">{medicine.get("mechanism", "Not specified")}</span>
      </div>
    </div>
    """

    with st.expander("⚡ Usage and Action"):
        st.markdown(content, unsafe_allow_html=True)


def render_dosage(medicine: dict) -> None:
    """
    Render the 'Dosage and Use' section.

    Fields: dosage, timing, admin_tips, spacing.

    Args:
        medicine (dict): Full medicine data dict.
    """
    content = f"""
    <div class="info-card">
      <div class="info-card-title">💧 DOSAGE AND USE</div>
      {_info_row("Recommended Dose", medicine.get("dosage", "Not specified"))}
      {_info_row("When to Take", medicine.get("timing", "Not specified"))}
      {_info_row("Dosing Interval", medicine.get("spacing", "Not specified"))}
      {_info_row("Administration Tips", medicine.get("admin_tips", "Not specified"))}
    </div>
    """

    with st.expander("💧 Dosage and Use"):
        st.markdown(content, unsafe_allow_html=True)


def render_warnings(medicine: dict) -> None:
    """
    Render the 'Warnings' section with amber styling.

    Fields: warning_pregnancy, warning_pediatric, warning_driving, warning_storage.

    Args:
        medicine (dict): Full medicine data dict.
    """
    warnings_list = [
        ("🤰 Pregnancy:", medicine.get("warning_pregnancy", "Not specified")),
        ("👶 Pediatric Use:", medicine.get("warning_pediatric", "Not specified")),
        ("🚗 Driving & Machinery:", medicine.get("warning_driving", "Not specified")),
        ("🏪 Storage:", medicine.get("warning_storage", "Not specified")),
    ]

    bullets_html = "".join(
        f'<div class="bullet-item">'
        f'<div class="bullet-dot warn"></div>'
        f'<span style="font-family:var(--font-ui);font-size:0.88rem;color:var(--text);line-height:1.55;">'
        f'<strong style="color:var(--warn);">{label}</strong> {text}'
        f'</span>'
        f'</div>'
        for label, text in warnings_list
        if text and text != "Not specified"
    )

    content = f"""
    <div class="info-card" style="border-color:rgba(245,158,11,0.25);background:rgba(245,158,11,0.04);">
      <div class="info-card-title" style="color:var(--warn);">⚠️ &nbsp; WARNINGS AND PRECAUTIONS</div>
      {bullets_html}
    </div>
    """

    with st.expander("⚠️  Warnings"):
        st.markdown(content, unsafe_allow_html=True)


def render_contraindications(medicine: dict) -> None:
    """
    Render the 'Contraindications' section with red styling.

    Fields: contraindications list.

    Args:
        medicine (dict): Full medicine data dict.
    """
    contra = medicine.get("contraindications", [])
    contra_html = _bullet_list(contra, dot_class="danger")

    content = f"""
    <div class="info-card" style="border-color:rgba(239,68,68,0.2);background:rgba(239,68,68,0.04);">
      <div class="info-card-title" style="color:var(--danger);">🚫 &nbsp; CONTRAINDICATIONS</div>
      <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--muted);margin-bottom:0.8rem;">
        Do NOT use this medicine if any of the following apply:
      </div>
      {contra_html}
    </div>
    """

    with st.expander("🚫 Contraindications"):
        st.markdown(content, unsafe_allow_html=True)


def render_interactions(medicine: dict) -> None:
    """
    Render the 'Drug Interactions' section with purple styling.

    Fields: interactions list.

    Args:
        medicine (dict): Full medicine data dict.
    """
    interactions = medicine.get("interactions", [])
    inter_html = _bullet_list(interactions, dot_class="purple")

    content = f"""
    <div class="info-card" style="border-color:rgba(168,85,247,0.2);">
      <div class="info-card-title" style="color:var(--purple);">🔗 &nbsp; DRUG INTERACTIONS</div>
      <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--muted);margin-bottom:0.8rem;">
        Inform your doctor about all medicines, supplements, and foods:
      </div>
      {inter_html}
    </div>
    """

    with st.expander("🔗 Drug Interactions"):
        st.markdown(content, unsafe_allow_html=True)


def render_side_effects(medicine: dict) -> None:
    """
    Render the 'Side Effects' section with two sub-sections.

    Fields: side_effects_common (amber dots) and side_effects_serious (red dots).

    Args:
        medicine (dict): Full medicine data dict.
    """
    common_se = medicine.get("side_effects_common", [])
    serious_se = medicine.get("side_effects_serious", [])

    common_html = _bullet_list(common_se, dot_class="warn")
    serious_html = _bullet_list(serious_se, dot_class="danger")

    content = f"""
    <div class="info-card">
      <div class="info-card-title">🔴 &nbsp; POSSIBLE SIDE EFFECTS</div>
      <div style="margin-bottom:1rem;">
        <div class="info-card-title" style="color:var(--warn);margin-bottom:0.5rem;">
          🟡 COMMON (usually mild)
        </div>
        {common_html}
      </div>
      <div>
        <div class="info-card-title" style="color:var(--danger);margin-bottom:0.5rem;">
          🔴 SERIOUS (seek medical attention)
        </div>
        {serious_html}
      </div>
    </div>
    """

    with st.expander("🔴 Side Effects"):
        st.markdown(content, unsafe_allow_html=True)


def render_availability(medicine: dict) -> None:
    """
    Render the 'Availability and Substitutes' section.

    Fields: pack_sizes (shown as tags), substitutes (shown as tags).

    Args:
        medicine (dict): Full medicine data dict.
    """
    pack_sizes = medicine.get("pack_sizes", [])
    substitutes = medicine.get("substitutes", [])

    pack_html = _pack_tags(pack_sizes)
    sub_html = _pack_tags(substitutes)

    content = f"""
    <div class="info-card">
      <div class="info-card-title">🛍️ &nbsp; AVAILABILITY AND SUBSTITUTES</div>
      <div style="margin-bottom:1rem;">
        <div class="info-card-title" style="margin-bottom:0.5rem;">📦 PACK SIZES AVAILABLE</div>
        <div>{pack_html}</div>
      </div>
      <div>
        <div class="info-card-title" style="margin-bottom:0.5rem;">🔄 THERAPEUTIC SUBSTITUTES</div>
        <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);margin-bottom:0.5rem;">
          ⚠️ Always consult your doctor before switching medications.
        </div>
        <div>{sub_html}</div>
      </div>
    </div>
    """

    with st.expander("🛍️  Availability and Substitutes"):
        st.markdown(content, unsafe_allow_html=True)


def render_sources(medicine: dict) -> None:
    """
    Render the 'Medical Information Sources' section.

    Fields: sources list.

    Args:
        medicine (dict): Full medicine data dict.
    """
    sources = medicine.get("sources", [])
    sources_html = _bullet_list(sources)

    content = f"""
    <div class="info-card">
      <div class="info-card-title">📚 &nbsp; MEDICAL INFORMATION SOURCES</div>
      <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);margin-bottom:0.8rem;">
        Information compiled from these offline reference databases:
      </div>
      {sources_html}
    </div>
    """

    with st.expander("📚 Sources"):
        st.markdown(content, unsafe_allow_html=True)


def render_all_sections(medicine: dict) -> None:
    """
    Render ALL 9 information sections in the correct display order.

    This is the main export called by app.py after render_medicine_header().

    Args:
        medicine (dict): Full medicine data dict from database.
    """
    # Render sections in the order they appear in the MedScan+ app layout
    render_basic_info(medicine)
    render_usage_action(medicine)
    render_dosage(medicine)
    render_warnings(medicine)
    render_contraindications(medicine)
    render_interactions(medicine)
    render_side_effects(medicine)
    render_availability(medicine)
    render_sources(medicine)
