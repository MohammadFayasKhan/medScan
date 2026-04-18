"""
ui_styles.py
============
This module injects the custom CSS design system into the Streamlit app.

Why custom CSS?
  Streamlit's default styling is functional but generic. MedScan AI needed
  a dark, clinical aesthetic that communicates trust and precision.
  All styling is managed centrally here so changes only need to be made
  in one place.

Design system:
  - Colour palette: dark navy (#0a0f1e) background, cyan (#00d4ff) for
    interactive elements, orange (#ff6b35) for accents, green/amber/red
    for status indicators.
  - Typography: Syne (headings), DM Sans (body), JetBrains Mono (data/labels)
    — all loaded from Google Fonts.
  - Components: medicine cards, chat bubbles, strategy badges, comparison
    cards, sidebar status dots, info rows, bullet items, pack tags.
  - Animations: scan pulsing circle, button hover transitions, card hover lift.

Note: Streamlit re-runs the entire script on every interaction, so inject_styles()
is called at the top of app.py on every run. CSS injection via st.markdown with
unsafe_allow_html=True is the standard approach for custom Streamlit themes.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import streamlit as st


def get_styles() -> str:
    """
    Return the complete CSS string for MEDSCAN AI's dark UI.

    Includes:
      - @import: Google Fonts (Syne, JetBrains Mono, DM Sans)
      - :root: Design tokens (colors, fonts)
      - Global body + Streamlit overrides
      - Hero header styles
      - Medicine card (header + info sections)
      - Chat interface styles
      - Scan interface styles
      - Badge and pill styles
      - Comparison UI styles
      - All @keyframes animations

    Returns:
        str: Complete <style> HTML string for st.markdown() injection.
    """
    return """
<style>
/* ═══════════════════════════════════════════════════════════════════
   GOOGLE FONTS IMPORT
════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ═══════════════════════════════════════════════════════════════════
   CSS VARIABLES (DESIGN TOKENS)
════════════════════════════════════════════════════════════════════ */
:root {
  --bg:          #0a0f1e;
  --bg-card:     #0f1829;
  --bg-card2:    #131f35;
  --bg-input:    #0d1526;
  --accent:      #00d4ff;
  --accent-dim:  #0095b3;
  --accent2:     #ff6b35;
  --warn:        #f59e0b;
  --danger:      #ef4444;
  --ok:          #22c55e;
  --purple:      #a855f7;
  --text:        #e2e8f0;
  --text-sec:    #94a3b8;
  --muted:       #64748b;
  --border:      rgba(0,212,255,0.15);
  --border2:     rgba(255,255,255,0.06);
  --font-ui:     'DM Sans', sans-serif;
  --font-display:'Syne', sans-serif;
  --font-mono:   'JetBrains Mono', monospace;
}

/* ═══════════════════════════════════════════════════════════════════
   GLOBAL BODY + STREAMLIT OVERRIDES
════════════════════════════════════════════════════════════════════ */
html, body, [data-testid="stApp"] {
  background-color: var(--bg) !important;
  /* Subtle vertical column grid overlay for terminal aesthetic */
  background-image: repeating-linear-gradient(
    90deg,
    transparent, transparent 39px,
    rgba(255,255,255,0.018) 39px, rgba(255,255,255,0.018) 40px
  ) !important;
  color: var(--text) !important;
  font-family: var(--font-ui) !important;
}

/* Streamlit main block padding */
[data-testid="block-container"] {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}

/* Hide Streamlit default header and footer */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Sidebar background */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080d1a 0%, #0a0f1e 100%) !important;
  border-right: 1px solid var(--border) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--bg-input) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-family: var(--font-ui) !important;
  transition: border-color 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
  animation: borderGlow 1.5s ease infinite !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-dim)) !important;
  color: #000 !important;
  font-family: var(--font-mono) !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.08em !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.4rem !important;
  transition: all 0.25s ease !important;
  text-transform: uppercase !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(0,212,255,0.35) !important;
  background: linear-gradient(135deg, #1adbff, var(--accent)) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-card) !important;
  border-radius: 12px !important;
  padding: 0.2rem !important;
  border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.05em !important;
  color: var(--text-sec) !important;
  border-radius: 8px !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,149,179,0.15)) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border) !important;
}

/* Expander */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}

.streamlit-expanderContent {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
}

/* Select boxes and multiselect */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--bg-input) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

/* Progress / metric */
.stMetric {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}

/* ═══════════════════════════════════════════════════════════════════
   HERO HEADER
════════════════════════════════════════════════════════════════════ */
.hero {
  width: 100%;
  background: radial-gradient(ellipse 70% 50% at 50% 0%, rgba(0,212,255,0.09) 0%, transparent 60%);
  border-bottom: 1px solid var(--border);
  padding: 1.8rem 0 1.4rem 0;
  margin-bottom: 1.5rem;
  text-align: center;
  position: relative;
  animation: fadeUp 0.4s ease;
}

.logo-wrap {
  font-family: var(--font-display);
  font-size: 2.8rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1;
  display: inline-block;
}

.logo-med { color: var(--accent); }
.logo-scan { color: var(--accent2); }
.logo-ai { color: var(--text); }

.hero-subtitle {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  letter-spacing: 0.22em;
  color: var(--muted);
  margin-top: 0.4rem;
  text-transform: uppercase;
}

.antigravity-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.8rem;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(255,107,53,0.12));
  border: 1px solid rgba(0,212,255,0.3);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  font-weight: 600;
}

.badge-ag { color: var(--accent); }
.badge-build { color: var(--accent2); }

/* ═══════════════════════════════════════════════════════════════════
   MEDICINE HEADER CARD
════════════════════════════════════════════════════════════════════ */
.med-header {
  width: 100%;
  background: linear-gradient(135deg, #0f1829 0%, #101d30 50%, #0c1525 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.8rem 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 40px rgba(0,212,255,0.07), 0 4px 32px rgba(0,0,0,0.4);
  animation: fadeUp 0.35s ease;
}

.med-header::before {
  content: '';
  position: absolute;
  top: -30px; left: -30px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(0,212,255,0.12), transparent 60%);
  border-radius: 50%;
  pointer-events: none;
}

.med-name {
  font-family: var(--font-display);
  font-size: 2.2rem;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: -0.01em;
  line-height: 1.1;
  text-transform: capitalize;
}

.med-generic {
  font-family: var(--font-ui);
  font-size: 0.95rem;
  color: var(--text-sec);
  margin-top: 0.3rem;
}

.med-meta-row {
  display: flex;
  gap: 1.5rem;
  margin-top: 1rem;
  flex-wrap: wrap;
}

.med-meta-item {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.med-meta-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.med-meta-value {
  font-family: var(--font-ui);
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--text);
}

/* ═══════════════════════════════════════════════════════════════════
   CATEGORY BADGES
════════════════════════════════════════════════════════════════════ */
.cat-badge {
  display: inline-block;
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.badge-warn { background: rgba(245,158,11,0.15); color: var(--warn); border: 1px solid rgba(245,158,11,0.4); }
.badge-ok   { background: rgba(34,197,94,0.15);  color: var(--ok);   border: 1px solid rgba(34,197,94,0.4); }
.badge-danger { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.4); }
.badge-purple { background: rgba(168,85,247,0.15); color: var(--purple); border: 1px solid rgba(168,85,247,0.4); }
.badge-accent { background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.35); }

/* ═══════════════════════════════════════════════════════════════════
   INFO CARDS (SECTIONS)
════════════════════════════════════════════════════════════════════ */
.info-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin: 0.5rem 0;
  animation: fadeUp 0.3s ease;
}

.info-card-title {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.info-row {
  display: flex;
  padding: 0.45rem 0;
  border-bottom: 1px solid var(--border2);
  gap: 1rem;
}

.info-row:last-child { border-bottom: none; }

.info-label {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--muted);
  min-width: 130px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding-top: 0.1rem;
}

.info-value {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  color: var(--text);
  flex: 1;
  line-height: 1.55;
}

.bullet-item {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  padding: 0.3rem 0;
}

.bullet-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  margin-top: 0.45rem;
  flex-shrink: 0;
}

.bullet-dot.warn  { background: var(--warn); }
.bullet-dot.danger { background: var(--danger); }
.bullet-dot.ok    { background: var(--ok); }
.bullet-dot.purple { background: var(--purple); }

/* ═══════════════════════════════════════════════════════════════════
   SECTION HEADERS
════════════════════════════════════════════════════════════════════ */
.section-header {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  padding: 0.3rem 0 0.8rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
}

/* ═══════════════════════════════════════════════════════════════════
   SCAN INTERFACE
════════════════════════════════════════════════════════════════════ */
.scan-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem 0 2rem 0;
  gap: 1rem;
}

.scan-circle {
  width: 140px;
  height: 140px;
  border-radius: 50%;
  border: 2px solid var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 3rem;
  background: radial-gradient(circle, rgba(0,212,255,0.06), transparent 70%);
  animation: pulse 2.5s ease-in-out infinite;
}

.scan-title {
  font-family: var(--font-display);
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text);
  text-align: center;
}

.scan-subtitle {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-align: center;
}

/* ═══════════════════════════════════════════════════════════════════
   CHATBOT UI
════════════════════════════════════════════════════════════════════ */
.chat-wrap {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
  margin-top: 1.5rem;
}

.chat-title {
  background: linear-gradient(135deg, #0f1829, #11203a);
  border-bottom: 1px solid var(--border);
  padding: 1rem 1.4rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.chat-title-text {
  font-family: var(--font-display);
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
}

.chat-messages {
  max-height: 420px;
  overflow-y: auto;
  padding: 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}

.chat-messages::-webkit-scrollbar {
  width: 4px;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 4px;
}

.msg-bot {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 4px 14px 14px 14px;
  padding: 0.9rem 1.1rem;
  max-width: 88%;
  align-self: flex-start;
  animation: slideIn 0.2s ease;
}

.msg-user {
  background: linear-gradient(135deg, #005a80, #003d57);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 14px 4px 14px 14px;
  padding: 0.9rem 1.1rem;
  max-width: 88%;
  align-self: flex-end;
  animation: slideIn 0.2s ease;
}

.msg-sender {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--muted);
  margin-bottom: 0.3rem;
  letter-spacing: 0.08em;
}

.msg-text {
  font-family: var(--font-ui);
  font-size: 0.88rem;
  color: var(--text);
  line-height: 1.6;
}

.intent-badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.6rem;
  background: rgba(0,212,255,0.1);
  color: var(--accent);
  border: 1px solid rgba(0,212,255,0.2);
  margin-top: 0.4rem;
  letter-spacing: 0.06em;
}

/* Quick chips */
.quick-chip {
  display: inline-block;
  padding: 0.35rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-sec) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  cursor: pointer;
  transition: all 0.2s ease;
  margin: 0.2rem;
  white-space: nowrap;
  font-weight: 500 !important;
  text-transform: none !important;
  letter-spacing: 0.03em !important;
}

.quick-chip:hover {
  border-color: var(--accent);
  color: var(--accent) !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,212,255,0.15);
}

/* Medical disclaimer */
.disclaimer-card {
  background: rgba(245,158,11,0.06);
  border: 1px solid rgba(245,158,11,0.25);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-top: 1rem;
}

.disclaimer-text {
  font-family: var(--font-ui);
  font-size: 0.8rem;
  color: var(--warn);
  line-height: 1.55;
}

/* ═══════════════════════════════════════════════════════════════════
   SEARCH RESULT & SUGGESTION CHIPS
════════════════════════════════════════════════════════════════════ */
.strategy-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.2rem 0.7rem;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.08em;
  margin-left: 0.5rem;
}

.strategy-exact   { background: rgba(34,197,94,0.12);  color: var(--ok);     border: 1px solid rgba(34,197,94,0.3); }
.strategy-tfidf   { background: rgba(0,212,255,0.12);  color: var(--accent); border: 1px solid rgba(0,212,255,0.3); }
.strategy-fuzzy   { background: rgba(245,158,11,0.12); color: var(--warn);   border: 1px solid rgba(245,158,11,0.3); }

/* ═══════════════════════════════════════════════════════════════════
   DATABASE GRID CARDS
════════════════════════════════════════════════════════════════════ */
.db-card {
  background: var(--bg-card);
  border: 1px solid var(--border2);
  border-radius: 14px;
  padding: 1.1rem;
  cursor: pointer;
  transition: all 0.22s ease;
  position: relative;
  overflow: hidden;
  animation: fadeUp 0.3s ease;
}

.db-card:hover {
  border-color: var(--border);
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.db-card-accent {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  border-radius: 14px 0 0 14px;
}

.db-card-name {
  font-family: var(--font-display);
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
  text-transform: capitalize;
  margin-bottom: 0.2rem;
  padding-left: 0.8rem;
}

.db-card-detail {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--muted);
  padding-left: 0.8rem;
  letter-spacing: 0.04em;
}

/* ═══════════════════════════════════════════════════════════════════
   COMPARISON UI
════════════════════════════════════════════════════════════════════ */
.compare-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem;
  animation: fadeUp 0.3s ease;
}

.metric-card {
  background: var(--bg-card2);
  border: 1px solid var(--border2);
  border-radius: 14px;
  padding: 1.2rem;
  text-align: center;
}

.metric-value {
  font-family: var(--font-display);
  font-size: 2.8rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
}

.metric-label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 0.3rem;
}

/* ═══════════════════════════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════════════════════════════ */
.sidebar-logo {
  font-family: var(--font-display);
  font-size: 1.4rem;
  font-weight: 800;
  text-align: center;
  padding: 0.8rem 0;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.35rem 0;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-sec);
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--ok);
  flex-shrink: 0;
}

.recent-pill {
  display: inline-block;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  background: rgba(0,212,255,0.08);
  border: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-sec);
  cursor: pointer;
  transition: all 0.2s ease;
  margin: 0.15rem;
}

.recent-pill:hover {
  background: rgba(0,212,255,0.15);
  border-color: var(--accent);
  color: var(--accent);
}

/* ═══════════════════════════════════════════════════════════════════
   PACK SIZE TAGS
════════════════════════════════════════════════════════════════════ */
.pack-tag {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border2);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-sec);
  margin: 0.15rem;
}

/* ═══════════════════════════════════════════════════════════════════
   LOADING SPINNER
════════════════════════════════════════════════════════════════════ */
.loading-spinner {
  display: inline-block;
  width: 20px; height: 20px;
  border: 2px solid rgba(0,212,255,0.2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* ═══════════════════════════════════════════════════════════════════
   TOAST NOTIFICATION
════════════════════════════════════════════════════════════════════ */
.toast {
  position: fixed;
  top: 1.2rem; right: 1.5rem;
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.8rem 1.2rem;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text);
  z-index: 9999;
  animation: slideIn 0.3s ease;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ═══════════════════════════════════════════════════════════════════
   @KEYFRAMES — ALL ANIMATIONS
════════════════════════════════════════════════════════════════════ */

/* Pulsing glow for the scan circle */
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(0,212,255,0.4); }
  50%  { box-shadow: 0 0 0 22px rgba(0,212,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
}

/* Fade in from below — used for cards and result sections */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Slide in from left — used for chat message bubbles */
@keyframes slideIn {
  from { opacity: 0; transform: translateX(-20px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* Loading skeleton shimmer effect */
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}

/* Loading spinner rotation */
@keyframes spin {
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Border glow pulse for focused input fields */
@keyframes borderGlow {
  0%, 100% { box-shadow: 0 0 0 2px rgba(0,212,255,0.15); }
  50%       { box-shadow: 0 0 0 3px rgba(0,212,255,0.28); }
}

/* ═══════════════════════════════════════════════════════════════════
   OCR RESULT DISPLAY
════════════════════════════════════════════════════════════════════ */
.ocr-candidate-pill {
  display: inline-block;
  padding: 0.3rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(0,212,255,0.07);
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--accent);
  margin: 0.2rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.ocr-candidate-pill.best {
  background: rgba(0,212,255,0.18);
  border-color: var(--accent);
  font-weight: 700;
}

.ocr-candidate-pill:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,212,255,0.2);
}

/* Not found state */
.not-found-box {
  background: rgba(239,68,68,0.05);
  border: 1px solid rgba(239,68,68,0.2);
  border-radius: 14px;
  padding: 1.5rem;
  text-align: center;
  animation: fadeUp 0.3s ease;
}

.not-found-title {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--danger);
  margin-bottom: 0.5rem;
}

</style>
"""


def inject_styles() -> None:
    """Inject all global CSS styles into the Streamlit page."""
    st.markdown(get_styles(), unsafe_allow_html=True)
