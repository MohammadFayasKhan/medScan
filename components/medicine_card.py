"""
medicine_card.py
================
This component renders the large medicine identification card shown at the
top of every search result.

Layout:
  ┌───────────────────────────────────────────────────┐
  │  💊  [CATEGORY BADGE]                             │
  │  MEDICINE NAME (large, cyan)                      │
  │  Generic Name  ·  Form                            │
  │  [Strength]  [Active Substance]  [Manufacturer]   │
  └───────────────────────────────────────────────────┘

Design decisions:
  - The category badge colour is determined by matching the category string
    against the CATEGORY_COLOR_MAP dictionary (substring matching so partial
    categories like "NSAID / Analgesic" still match correctly).
  - The medicine name is displayed in large Syne font to make it immediately
    recognisable.
  - The search strategy badge (Exact / TF-IDF / Fuzzy) and confidence score
    are shown above the card so the user knows how the medicine was found.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import streamlit as st


# ─────────────────────────────────────────────────────────────────────
# CATEGORY → BADGE COLOR MAP
# ─────────────────────────────────────────────────────────────────────

CATEGORY_COLOR_MAP = {
    "analgesic":           ("badge-ok",     "✓"),
    "antipyretic":         ("badge-ok",     "✓"),
    "antibiotic":          ("badge-ok",     "✓ RX"),
    "nsaid":               ("badge-warn",   "⚠"),
    "antidiabetic":        ("badge-purple", "Rx"),
    "biguanide":           ("badge-purple", "Rx"),
    "ophthalmic":          ("badge-accent", "👁"),
    "antihistamine":       ("badge-accent", "H₁"),
    "ppi":                 ("badge-warn",   "Rx"),
    "proton pump":         ("badge-warn",   "Rx"),
    "antitussive":         ("badge-ok",     "OTC"),
    "antihypertensive":    ("badge-danger", "Rx"),
    "calcium channel":     ("badge-danger", "Rx"),
    "statin":              ("badge-purple", "Rx"),
    "lipid":               ("badge-purple", "Rx"),
    "bronchodilator":      ("badge-accent", "Rx"),
    "beta-2":              ("badge-accent", "Rx"),
    "antiepileptic":       ("badge-danger", "Rx"),
    "benzodiazepine":      ("badge-danger", "C-IV"),
    "arb":                 ("badge-warn",   "Rx"),
    "angiotensin":         ("badge-warn",   "Rx"),
    "antiplatelet":        ("badge-danger", "Rx"),
    "leukotriene":         ("badge-ok",     "Rx"),
    "vitamin":             ("badge-ok",     "OTC"),
    "supplement":          ("badge-ok",     "OTC"),
}


def get_category_badge(category: str) -> tuple:
    """
    Map a medicine category string to a CSS badge class and label.

    Performs case-insensitive substring matching so partial categories
    like "NSAID / Analgesic / Antipyretic" correctly map to "badge-warn".

    Args:
        category (str): Medicine category string from database.

    Returns:
        tuple[str, str]: (css_class, short_label).

    Example:
        >>> cls, label = get_category_badge("NSAID / Analgesic")
        >>> cls
        'badge-warn'
    """
    cat_lower = category.lower()

    for key, (css_class, label) in CATEGORY_COLOR_MAP.items():
        if key in cat_lower:
            return css_class, label

    # Default: cyan accent badge
    return "badge-accent", "Rx"


def render_medicine_header(medicine: dict) -> None:
    """
    Render the medicine identification header card with custom HTML/CSS.

    Layout:
      ┌─────────────────────────────────────────────────┐
      │  💊  [CATEGORY BADGE]                           │
      │  MEDICINE NAME (large, cyan)                    │
      │  Generic Name · Form                            │
      │  [Strength]  [Category]  [Active Substance]     │
      └─────────────────────────────────────────────────┘

    Grid glow in top-left corner, Syne font for name, DM Sans for detail.

    Args:
        medicine (dict): Full medicine data dict from database.
    """
    name = medicine.get("name", "Unknown").title()
    generic = medicine.get("generic_name", "")
    category = medicine.get("category", "Unknown")
    strength = medicine.get("strength", "N/A")
    form = medicine.get("form", "N/A")
    active = medicine.get("active_substance", "N/A")
    manufacturer = medicine.get("manufacturer", "")

    badge_class, badge_label = get_category_badge(category)

    # Build the header card HTML
    header_html = f"""
    <div class="med-header">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;">
        <div>
          <div style="margin-bottom:0.6rem;">
            <span style="font-size:2rem;">💊</span>&nbsp;
            <span class="cat-badge {badge_class}">{badge_label}&nbsp;{category.split('/')[0].strip()}</span>
          </div>
          <div class="med-name">{name}</div>
          <div class="med-generic">{generic} &nbsp;·&nbsp; {form}</div>
        </div>
      </div>
      <div class="med-meta-row">
        <div class="med-meta-item">
          <span class="med-meta-label">Strength</span>
          <span class="med-meta-value">{strength}</span>
        </div>
        <div class="med-meta-item">
          <span class="med-meta-label">Active Substance</span>
          <span class="med-meta-value">{active.split("(")[0].strip()}</span>
        </div>
        {"<div class='med-meta-item'><span class='med-meta-label'>Manufacturer</span><span class='med-meta-value'>" + manufacturer.split("/")[0].strip() + "</span></div>" if manufacturer and manufacturer != "Not specified" else ""}
      </div>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)


def render_search_strategy_badge(strategy: str, confidence: float,
                                  search_time_ms: float) -> None:
    """
    Render a small badge showing which search strategy matched the medicine.

    Shows: strategy name (exact/tfidf/fuzzy), confidence %, and search time.

    Args:
        strategy (str): Strategy used: "exact", "tfidf", "fuzzy", or "none".
        confidence (float): Confidence score 0.0–1.0.
        search_time_ms (float): Search duration in milliseconds.
    """
    strategy_labels = {
        "exact": ("strategy-exact", "✓ EXACT MATCH", "100%"),
        "tfidf": ("strategy-tfidf", "~ TF-IDF MATCH", f"{confidence*100:.0f}%"),
        "fuzzy": ("strategy-fuzzy", "≈ FUZZY MATCH", f"{confidence*100:.0f}%"),
        "none":  ("strategy-exact", "NOT FOUND", "—"),
    }
    css_class, label, conf_str = strategy_labels.get(strategy, ("strategy-tfidf", strategy.upper(), "—"))

    st.markdown(
        f'<div style="margin-bottom:0.5rem;">'
        f'<span class="strategy-badge {css_class}">{label} &nbsp; {conf_str}</span>'
        f'<span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--muted);margin-left:0.5rem;">'
        f'⚡ {search_time_ms:.0f}ms</span>'
        f'</div>',
        unsafe_allow_html=True
    )
