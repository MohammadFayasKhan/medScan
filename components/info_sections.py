"""
info_sections.py
================
Renders the collapsible accordions specifically tuned for the 11k Medicine details database.

Columns rendered:
  - Image URL
  - Composition
  - Manufacturer
  - Uses
  - Side_effects
  - Excellent Review %
  - Average Review %
  - Poor Review %
"""

import streamlit as st

def render_all_sections(medicine: dict) -> None:
    """
    Render all collapsible information sections for the medicine.
    Tailored for the 11k Medicine Details dataset.
    """
    
    # Render Image if available
    img_url = str(medicine.get("Image URL", ""))
    if img_url and img_url.startswith("http"):
        # Put image in a nice centered card
        st.markdown(f'''
            <div style="display:flex; justify-content:center; align-items:center; background:var(--bg-card); 
                 border:1px solid var(--border); border-radius:14px; padding:2rem; margin-bottom:1rem;">
                <img src="{img_url}" style="max-height: 250px; max-width: 100%; border-radius: 8px;">
            </div>
            ''', unsafe_allow_html=True)
            
    # Basic Information (Composition, Manufacturer)
    with st.expander("ℹ️ Basic Information", expanded=True):
        composition = str(medicine.get("Composition", "Not specified"))
        manufacturer = str(medicine.get("Manufacturer", "Not specified"))
        st.markdown(f'''
        <div class="info-card">
            <div class="info-row">
                <span class="info-label">Composition</span>
                <span class="info-value">{composition if composition != 'nan' else 'Not specified'}</span>
            </div>
            <div class="info-row" style="border-bottom:none">
                <span class="info-label">Manufacturer</span>
                <span class="info-value">{manufacturer if manufacturer != 'nan' else 'Not specified'}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Uses
    uses = str(medicine.get("Uses", ""))
    if uses and uses != "Not specified" and uses != 'nan':
        with st.expander("🩺 Uses and Indications", expanded=False):
            uses_list = [u.strip() for u in uses.split(" ") if len(u.strip()) > 3]
            bullets = "".join([f'<div class="bullet-item"><div class="bullet-dot"></div><div class="info-value" style="font-family:var(--font-ui)">{u}</div></div>' for u in uses_list[:10]])
            if not bullets or len(uses_list) < 3:
                bullets = f'<div class="info-value" style="font-family:var(--font-ui)">{uses}</div>'
            st.markdown(f'<div class="info-card">{bullets}</div>', unsafe_allow_html=True)

    # Side Effects
    side_effects = str(medicine.get("Side_effects", ""))
    if side_effects and side_effects != "Not specified" and side_effects != 'nan':
        with st.expander("⚠️ Possible Side Effects", expanded=False):
            se_list = [s.strip() for s in side_effects.split(" ") if len(s.strip()) > 3]
            bullets = "".join([f'<div class="bullet-item"><div class="bullet-dot warn"></div><div class="info-value" style="font-family:var(--font-ui)">{s}</div></div>' for s in se_list[:10]])
            if not bullets or len(se_list) < 3:
                bullets = f'<div class="info-value" style="font-family:var(--font-ui)">{side_effects}</div>'
            st.markdown(f'<div class="info-card" style="border-color:rgba(245,158,11,0.25);background:rgba(245,158,11,0.04);">{bullets}</div>', unsafe_allow_html=True)

    # Patient Reviews
    review_exc = str(medicine.get("Excellent Review %", "0")).replace("%","")
    if review_exc != 'nan':
        with st.expander("⭐ Patient Reviews", expanded=False):
            exc = str(medicine.get("Excellent Review %", "0")).replace("%","")
            avg = str(medicine.get("Average Review %", "0")).replace("%","")
            poor = str(medicine.get("Poor Review %", "0")).replace("%","")
            
            # Handle NaNs from dataset
            exc = exc if exc != 'nan' else '0'
            avg = avg if avg != 'nan' else '0'
            poor = poor if poor != 'nan' else '0'
            
            st.markdown(f'''
            <div class="info-card">
            <div class="info-row">
                <span class="info-label" style="color:var(--ok)">Excellent</span>
                <div style="flex:1; display:flex; align-items:center;">
                    <div style="width:100%; background:rgba(255,255,255,0.05); height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{exc}%; background:var(--ok); height:100%;"></div>
                    </div>
                    <span class="info-value" style="margin-left:1rem; min-width:40px">{exc}%</span>
                </div>
            </div>
            <div class="info-row">
                <span class="info-label" style="color:var(--warn)">Average</span>
                <div style="flex:1; display:flex; align-items:center;">
                    <div style="width:100%; background:rgba(255,255,255,0.05); height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{avg}%; background:var(--warn); height:100%;"></div>
                    </div>
                    <span class="info-value" style="margin-left:1rem; min-width:40px">{avg}%</span>
                </div>
            </div>
            <div class="info-row" style="border-bottom:none">
                <span class="info-label" style="color:var(--danger)">Poor</span>
                <div style="flex:1; display:flex; align-items:center;">
                    <div style="width:100%; background:rgba(255,255,255,0.05); height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{poor}%; background:var(--danger); height:100%;"></div>
                    </div>
                    <span class="info-value" style="margin-left:1rem; min-width:40px">{poor}%</span>
                </div>
            </div>
            </div>
            ''', unsafe_allow_html=True)
