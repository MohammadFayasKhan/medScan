"""
medicine_db.py
==============
This module is responsible for loading and querying the offline medicine database.
The database is a CSV file stored at data/medicines.csv — no internet needed.

What this module does:
  - Reads medicines.csv into a pandas DataFrame at startup and caches it
    using Streamlit's @st.cache_data so repeated calls don't re-read the disk.
  - Validates that all 27 required columns are present in the CSV.
  - Converts columns like "brand_names" and "indications" from comma-separated
    strings into actual Python lists, which makes them much easier to display.
  - Provides helper functions for name/category lookup used by the search engine
    and the comparison engine.
  - Builds the text corpus that the TF-IDF search index is trained on.

Design decision:
  We chose CSV over SQLite because it is human-readable, easy to edit, and
  portable. For 20 medicines, the performance difference is negligible.
  The pandas + cache_data combination gives us the performance of an in-memory
  database without the setup complexity.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import os
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Default path to medicine database CSV relative to project root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medicines.csv")

# All columns that MUST exist in the CSV for the app to function
REQUIRED_COLUMNS = [
    "name", "generic_name", "brand_names", "form", "strength",
    "category", "active_substance", "manufacturer", "features",
    "uses", "mechanism", "indications", "dosage", "timing",
    "admin_tips", "spacing", "warning_pregnancy", "warning_pediatric",
    "warning_driving", "warning_storage", "contraindications",
    "interactions", "side_effects_common", "side_effects_serious",
    "substitutes", "pack_sizes", "sources"
]

# Columns stored as comma-separated strings in CSV → convert to Python lists
LIST_COLUMNS = [
    "brand_names", "features", "contraindications", "interactions",
    "side_effects_common", "side_effects_serious", "substitutes",
    "pack_sizes", "sources", "indications"
]

# Category → display color mapping (used by card renderer)
CATEGORY_COLORS = {
    "analgesic": "#22c55e",
    "antibiotic": "#22c55e",
    "nsaid": "#f59e0b",
    "antidiabetic": "#a855f7",
    "ophthalmic": "#00d4ff",
    "antihistamine": "#3b82f6",
    "ppi": "#ff6b35",
    "antitussive": "#94a3b8",
    "antihypertensive": "#ef4444",
    "statin": "#ec4899",
    "bronchodilator": "#06b6d4",
    "antiepileptic": "#8b5cf6",
    "arb": "#f97316",
    "antiplatelet": "#dc2626",
    "leukotriene": "#10b981",
    "vitamin": "#fbbf24",
    "biguanide": "#a855f7",
    "calcium channel blocker": "#ef4444",
    "benzodiazepine": "#7c3aed",
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# DATABASE LOADING
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_database(csv_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load medicine CSV database into a validated pandas DataFrame.

    Uses Streamlit's @st.cache_data to load only once per session,
    preventing repeated disk I/O on every rerun.

    Processing steps:
      1. Verify file exists on disk
      2. Read CSV with UTF-8 encoding
      3. Validate all required columns are present
      4. Fill NaN values with "Not specified" placeholder
      5. Convert LIST_COLUMNS from comma-separated strings to Python lists
      6. Add 'name_lower' column for efficient case-insensitive search

    Args:
        csv_path (str): Path to medicines.csv. Defaults to DB_PATH constant.

    Returns:
        pd.DataFrame: Validated and processed medicine DataFrame.

    Raises:
        FileNotFoundError: If CSV file not found at csv_path.
        ValueError: If required columns are missing from the CSV.

    Example:
        >>> df = load_database()
        >>> len(df) >= 20
        True
    """
    # Step 1: Verify the file exists before attempting read
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Medicine database not found at: {csv_path}\n"
            "Please ensure data/medicines.csv exists. Run setup.py if needed."
        )

    # Step 2: Load CSV with explicit UTF-8 encoding (handles special chars)
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Step 3: Validate that all required columns are present
    valid, missing = validate_database(df)
    if not valid:
        raise ValueError(
            f"medicines.csv is missing required columns: {missing}\n"
            "Please check DATASET_GUIDE.md for the correct CSV schema."
        )

    # Step 4: Replace all NaN cells with "Not specified" string placeholder
    # This prevents KeyError/None issues when rendering fields in the UI
    df = df.fillna("Not specified")

    # Step 5: Convert comma-separated list columns into actual Python lists
    # Example: "Crocin, Dolo, Calpol" → ["Crocin", "Dolo", "Calpol"]
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: [item.strip() for item in str(x).split(",")]
                if pd.notna(x) and str(x) != "Not specified"
                else []
            )

    # Step 6: Add lowercase name column for O(1) case-insensitive lookups
    df["name_lower"] = df["name"].str.lower().str.strip()

    logger.info(f"Database loaded: {len(df)} medicines, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────

def validate_database(df: pd.DataFrame) -> tuple:
    """
    Check that all required columns exist and data is non-empty.

    Args:
        df (pd.DataFrame): DataFrame loaded from CSV.

    Returns:
        tuple[bool, list[str]]: (is_valid, missing_columns_list).
            If valid: (True, []).
            If invalid: (False, ["missing_col1", "missing_col2", ...]).

    Example:
        >>> valid, missing = validate_database(df)
        >>> assert valid == True
    """
    # Check DataFrame is not empty
    if df.empty:
        return False, ["<DataFrame is empty — no data rows>"]

    # Identify any columns required by the app that are absent in the CSV
    existing = set(df.columns.str.lower().tolist())
    required = set(REQUIRED_COLUMNS)
    missing = list(required - existing)

    # Return validation result
    if missing:
        return False, sorted(missing)
    return True, []


# ─────────────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def get_medicine_by_key(df: pd.DataFrame, key: str) -> dict:
    """
    Get full medicine record by its primary key (lowercase name).

    Does an exact match on the 'name_lower' column which is guaranteed
    to exist after load_database() processes the DataFrame.

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.
        key (str): Lowercase medicine name (e.g., "paracetamol").

    Returns:
        dict | None: Complete medicine record as a Python dict,
                     or None if no match found.

    Example:
        >>> med = get_medicine_by_key(df, "paracetamol")
        >>> med["generic_name"]
        'Acetaminophen'
    """
    # Normalise key: lowercase and strip surrounding whitespace
    key = key.lower().strip()

    # Filter DataFrame to rows where name_lower matches exactly
    matches = df[df["name_lower"] == key]

    if matches.empty:
        # No exact match found
        return None

    # Return the first matching row as a Python dict
    return matches.iloc[0].to_dict()


def search_by_name(df: pd.DataFrame, query: str) -> dict:
    """
    Search across name, generic_name, and brand_names columns.

    Priority order (returns first match found):
      1. Exact match on name_lower
      2. name_lower starts with query
      3. name_lower contains query
      4. generic_name (lowercase) contains query
      5. brand_names string contains query

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.
        query (str): Search string (can be partial name or brand name).

    Returns:
        dict | None: First matching medicine record as dict, or None.

    Example:
        >>> med = search_by_name(df, "crocin")
        >>> med["name"]
        'paracetamol'
    """
    if not query:
        return None

    # Normalise query for case-insensitive comparison
    q = query.lower().strip()

    # Priority 1: Exact name match
    exact = df[df["name_lower"] == q]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # Priority 2: Name starts with query (prefix match)
    prefix = df[df["name_lower"].str.startswith(q)]
    if not prefix.empty:
        return prefix.iloc[0].to_dict()

    # Priority 3: Name contains query (substring match)
    contains_name = df[df["name_lower"].str.contains(q, na=False)]
    if not contains_name.empty:
        return contains_name.iloc[0].to_dict()

    # Priority 4: Generic name contains query
    contains_generic = df[df["generic_name"].str.lower().str.contains(q, na=False)]
    if not contains_generic.empty:
        return contains_generic.iloc[0].to_dict()

    # Priority 5: Brand names column contains query (brand name lookup)
    # brand_names is already a list after load_database processing
    for _, row in df.iterrows():
        # Convert brand list to lowercase string for substring search
        brands_str = ", ".join(row["brand_names"]).lower() if isinstance(row["brand_names"], list) else str(row["brand_names"]).lower()
        if q in brands_str:
            return row.to_dict()

    # No match found in any column
    return None


# ─────────────────────────────────────────────────────────────────────
# COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────

def get_all_names(df: pd.DataFrame) -> list:
    """
    Return a sorted list of all medicine display names (title case).

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.

    Returns:
        list[str]: Alphabetically sorted list of medicine names.

    Example:
        >>> names = get_all_names(df)
        >>> "paracetamol" in names
        True
    """
    # Return names as they appear in the 'name' column, sorted A-Z
    return sorted(df["name"].tolist())


def get_all_categories(df: pd.DataFrame) -> list:
    """
    Return a sorted list of unique medicine categories in the database.

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.

    Returns:
        list[str]: Sorted unique list of category strings.

    Example:
        >>> cats = get_all_categories(df)
        >>> "NSAID / Analgesic / Antipyretic" in cats
        True
    """
    # Get unique values from 'category' column sorted alphabetically
    return sorted(df["category"].unique().tolist())


def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Return sub-DataFrame filtered to medicines matching the given category.

    Performs case-insensitive substring match so partial category names work.

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.
        category (str): Category string to filter by (case-insensitive).

    Returns:
        pd.DataFrame: Filtered DataFrame with matching rows only.

    Example:
        >>> nsaids = filter_by_category(df, "nsaid")
        >>> len(nsaids) > 0
        True
    """
    if not category or category == "All":
        # Return complete DataFrame if no filter applied
        return df

    # Case-insensitive substring match on category
    return df[df["category"].str.lower().str.contains(category.lower(), na=False)]


def get_medicine_card_data(row: pd.Series) -> dict:
    """
    Extract minimal fields needed to render a medicine browser card.

    Returns only the fields shown in the database grid view,
    not the full medicine record (which is loaded on demand).

    Args:
        row (pd.Series): A single row from the medicine DataFrame.

    Returns:
        dict: Minimal card data with keys: name, generic_name,
              category, strength, form, color_key.

    Example:
        >>> card = get_medicine_card_data(df.iloc[0])
        >>> "name" in card
        True
    """
    # Determine display color key from category slug
    category_lower = str(row.get("category", "")).lower()
    color_key = "#00d4ff"  # default cyan accent

    # Find a matching color from the category color map
    for key, color in CATEGORY_COLORS.items():
        if key in category_lower:
            color_key = color
            break

    return {
        "name": row.get("name", "Unknown"),
        "generic_name": row.get("generic_name", "Unknown"),
        "category": row.get("category", "Unknown"),
        "strength": row.get("strength", "N/A"),
        "form": row.get("form", "N/A"),
        "color_key": color_key,
    }


# ─────────────────────────────────────────────────────────────────────
# SEARCH CORPUS BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_search_corpus(df: pd.DataFrame) -> list:
    """
    Build a text corpus for TF-IDF search index from the medicine database.

    Each document in the corpus corresponds to one medicine row.
    Combines all searchable text fields into one large searchable string.
    This corpus is passed to TfidfVectorizer in medicine_search.py.

    Fields included (in order of relevance weight):
      name (×3 repeated for boosting), generic_name, brand_names,
      active_substance, category, uses, indications

    Args:
        df (pd.DataFrame): Processed medicine DataFrame.

    Returns:
        list[str]: List of text documents, one per medicine row.

    Example:
        >>> corpus = build_search_corpus(df)
        >>> len(corpus) == len(df)
        True
    """
    corpus = []

    for _, row in df.iterrows():
        # Repeat name 3× to boost exact name matches in TF-IDF scores
        name_boost = f"{row['name']} {row['name']} {row['name']}"

        # Flatten brand_names list back to string for vectoriser
        brands = (
            " ".join(row["brand_names"])
            if isinstance(row["brand_names"], list)
            else str(row["brand_names"])
        )

        # Flatten indications list if present
        indications = (
            " ".join(row["indications"])
            if isinstance(row.get("indications", []), list)
            else str(row.get("indications", ""))
        )

        # Combine all searchable fields into one document string
        doc = " ".join([
            name_boost,
            str(row.get("generic_name", "")),
            brands,
            str(row.get("active_substance", "")),
            str(row.get("category", "")),
            str(row.get("uses", "")),
            indications,
        ])

        corpus.append(doc)

    return corpus
