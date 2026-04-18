"""
medicine_db.py
==============
This module is responsible for loading and querying the offline medicine database.
The database is a CSV file stored at data/Medicine_Details.csv — no internet needed.

What this module does:
  - Reads Medicine_Details.csv (11k+ rows) into a pandas DataFrame at startup.
  - Validates required columns.
  - Provides helper functions for name lookup used by the search engine.
  - Builds the text corpus that the TF-IDF search index is trained on.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 2.0.0
"""

import os
import logging
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Medicine_Details.csv")

REQUIRED_COLUMNS = [
    "Medicine Name", "Composition", "Uses", "Side_effects", 
    "Image URL", "Manufacturer", "Excellent Review %", 
    "Average Review %", "Poor Review %"
]

CATEGORY_COLORS = {
    "pain": "#22c55e",
    "infection": "#22c55e",
    "fever": "#f59e0b",
    "diabetes": "#a855f7",
    "eye": "#00d4ff",
    "allergy": "#3b82f6",
    "acid": "#ff6b35",
    "cough": "#94a3b8",
    "blood pressure": "#ef4444",
    "cholesterol": "#ec4899",
    "asthma": "#06b6d4",
    "cancer": "#8b5cf6",
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# DATABASE LOADING
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_database(csv_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load medicine CSV database into a validated pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Medicine database not found at: {csv_path}\n"
            "Please ensure data/Medicine_Details.csv exists."
        )

    # Read CSV
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Validate
    valid, missing = validate_database(df)
    if not valid:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Fill NaNs
    df = df.fillna("Not specified")

    # Clean & standardize core search fields
    # Standardize column names dynamically for UI
    df["name"] = df["Medicine Name"].astype(str)
    df["name_lower"] = df["name"].str.lower().str.strip()
    df["generic_name"] = df["Composition"].astype(str)
    
    # Pre-calculate a mock category based on uses for color coding
    def guess_category(uses_str):
        u = str(uses_str).lower()
        for k in CATEGORY_COLORS.keys():
            if k in u:
                return k.title()
        return "General"
        
    df["category"] = df["Uses"].apply(guess_category)

    logger.info(f"Database loaded: {len(df)} medicines, {len(df.columns)} columns")
    return df


def validate_database(df: pd.DataFrame) -> tuple:
    if df.empty:
        return False, ["<DataFrame is empty>"]
    
    existing = set(df.columns.tolist())
    required = set(REQUIRED_COLUMNS)
    missing = list(required - existing)
    
    if missing:
        return False, sorted(missing)
    return True, []


# ─────────────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def get_medicine_by_key(df: pd.DataFrame, key: str) -> dict:
    key = key.lower().strip()
    matches = df[df["name_lower"] == key]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def search_by_name(df: pd.DataFrame, query: str) -> dict:
    if not query:
        return None
    
    q = query.lower().strip()

    # Exact match
    exact = df[df["name_lower"] == q]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # Prefix
    prefix = df[df["name_lower"].str.startswith(q)]
    if not prefix.empty:
        return prefix.iloc[0].to_dict()

    # Substring
    contains_name = df[df["name_lower"].str.contains(q, na=False, regex=False)]
    if not contains_name.empty:
        return contains_name.iloc[0].to_dict()

    # Composition
    contains_comp = df[df["generic_name"].str.lower().str.contains(q, na=False, regex=False)]
    if not contains_comp.empty:
        return contains_comp.iloc[0].to_dict()

    return None


def get_all_names(df: pd.DataFrame) -> list:
    return sorted(df["name"].tolist())


def get_all_categories(df: pd.DataFrame) -> list:
    return sorted(df["category"].unique().tolist())


def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    if not category or category == "All":
        return df
    return df[df["category"].str.lower() == category.lower()]


def get_medicine_card_data(row: pd.Series) -> dict:
    cat = str(row.get("category", "")).lower()
    color_key = CATEGORY_COLORS.get(cat, "#00d4ff")

    return {
        "name": row.get("name", "Unknown"),
        "generic_name": row.get("generic_name", "Unknown"),
        "category": row.get("category", "Unknown"),
        "strength": "Varies by composition",
        "form": "N/A",
        "color_key": color_key,
    }

# ─────────────────────────────────────────────────────────────────────
# SEARCH CORPUS BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_search_corpus(df: pd.DataFrame) -> list:
    """
    Build a text corpus for TF-IDF search index from the medicine database.
    Since this is 11k rows, we focus on the most important fields to keep
    vectorization fast.
    """
    corpus = []
    
    # We use numpy arrays for faster iteration over 11k rows
    names = df['name'].values
    comps = df['generic_name'].values
    uses = df['Uses'].values
    
    for i in range(len(df)):
        n = str(names[i])
        # Name repeated twice to boost TF-IDF exact hits
        doc = f"{n} {n} {comps[i]} {uses[i]}"
        corpus.append(doc)
        
    return corpus

