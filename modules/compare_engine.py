"""
compare_engine.py
=================
This module handles side-by-side comparison of two or three medicines.

What the comparison does:
  Each medicine is scored across several dimensions derived from its
  database fields. The scores (0-100) are then:
  - Displayed as a radar (spider) chart using matplotlib
  - Used to generate a plain-English verdict explaining which medicine
    performs better in each dimension and for what patient types

Scoring dimensions:
  - Safety Score     : based on severity of side effects and contraindications
  - Avail. Score     : number of pack sizes and substitute options
  - Ease of Use      : dosing frequency and administration complexity
  - Info Quality     : completeness of the database record
  - Profile Score    : interaction count and warning severity

Design note:
  The radar chart uses matplotlib with a dark background to match the app's
  colour scheme. The chart is rendered as a matplotlib Figure object and
  displayed using st.pyplot() in the compare_ui.py component.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Dimensions compared and their display labels
COMPARISON_DIMENSIONS = [
    "safety_score",
    "side_effect_count",
    "interaction_count",
    "substitute_count",
    "warning_count",
]

DIMENSION_LABELS = {
    "safety_score": "Safety",
    "side_effect_count": "Side Effects",
    "interaction_count": "Interactions",
    "substitute_count": "Alternatives",
    "warning_count": "Warnings",
}

# Colors for radar chart — one per medicine (up to 3)
PROCESS_COLORS = ["#00d4ff", "#ff6b35", "#a855f7"]

# Dark theme colors matching the app UI
CHART_BG = "#0f1829"
CHART_GRID = "rgba(255,255,255,0.1)"

# Fields shown in the comparison table
COMPARE_TABLE_FIELDS = [
    ("Category", "category"),
    ("Form", "form"),
    ("Strength", "strength"),
    ("Uses", "uses"),
    ("Dosage", "dosage"),
    ("Timing", "timing"),
    ("Common Side Effects", "side_effects_common"),
    ("Serious Side Effects", "side_effects_serious"),
    ("Contraindications", "contraindications"),
    ("Pregnancy Warning", "warning_pregnancy"),
    ("Pediatric Use", "warning_pediatric"),
]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────

def compute_medicine_scores(medicine: dict) -> dict:
    """
    Compute numerical comparison scores for a single medicine dict.

    Scores are designed so that higher = better (except where noted).

    Dimensions:
      safety_score       = 10 - num_contraindications (higher = safer)
      side_effect_count  = num common + serious side effects (lower = better in raw)
      interaction_count  = num known interactions (lower = better in raw)
      substitute_count   = num substitutes available (higher = more options)
      warning_count      = num warning fields with real content (lower = better)

    Note: For radar chart display, all inverted dimensions are normalised
    in generate_radar_chart() so that outward = better in all directions.

    Args:
        medicine (dict): Full medicine data dict from database.

    Returns:
        dict: Scores keyed by COMPARISON_DIMENSIONS strings.

    Example:
        >>> scores = compute_medicine_scores(paracetamol_dict)
        >>> scores["safety_score"] > 0
        True
    """
    def list_len(val) -> int:
        """Safely count list items from a field that may be list or string."""
        if isinstance(val, list):
            return len([v for v in val if v and v.strip()])
        if isinstance(val, str) and val not in ("Not specified", ""):
            return len([v for v in val.split(",") if v.strip()])
        return 0

    # Count contraindications — more = less safe
    n_contra = list_len(medicine.get("contraindications", []))

    # Safety score: base 10 minus number of contraindications (min 0)
    safety_score = max(0, 10 - n_contra)

    # Count total side effects (common + serious)
    n_common_se = list_len(medicine.get("side_effects_common", []))
    n_serious_se = list_len(medicine.get("side_effects_serious", []))
    side_effect_count = n_common_se + n_serious_se

    # Count known drug interactions
    interaction_count = list_len(medicine.get("interactions", []))

    # Count available substitutes
    substitute_count = list_len(medicine.get("substitutes", []))

    # Count warning fields with real (non-placeholder) content
    warning_fields = [
        "warning_pregnancy", "warning_pediatric",
        "warning_driving", "warning_storage"
    ]
    warning_count = sum(
        1 for f in warning_fields
        if medicine.get(f, "Not specified") not in ("Not specified", "")
    )

    return {
        "safety_score": safety_score,
        "side_effect_count": side_effect_count,
        "interaction_count": interaction_count,
        "substitute_count": substitute_count,
        "warning_count": warning_count,
    }


# ─────────────────────────────────────────────────────────────────────
# COMPARISON LOGIC
# ─────────────────────────────────────────────────────────────────────

def compare_medicines(medicines: list) -> dict:
    """
    Compare 2–3 medicines across all defined scoring dimensions.

    Computes scores for each medicine and determines which medicine
    "wins" each dimension (best score on that metric).

    Args:
        medicines (list[dict]): List of 2 or 3 medicine data dicts.

    Returns:
        dict: Comparison result containing:
            {
              "scores": list[dict],      # score dict per medicine
              "winners": dict,           # {dimension: winning medicine name}
              "medicines": list[dict]    # input medicines (for reference)
            }

    Raises:
        ValueError: If fewer than 2 or more than 3 medicines provided.

    Example:
        >>> result = compare_medicines([paracetamol, ibuprofen])
        >>> result["winners"]["safety_score"]
        'paracetamol'
    """
    if len(medicines) < 2:
        raise ValueError("Comparison requires at least 2 medicines.")
    if len(medicines) > 3:
        raise ValueError("Maximum 3 medicines can be compared at once.")

    # Compute scores for each medicine
    all_scores = [compute_medicine_scores(m) for m in medicines]

    # Determine winner for each dimension
    # Winner = medicine with best score on that dimension
    winners = {}
    for dim in COMPARISON_DIMENSIONS:
        values = [scores[dim] for scores in all_scores]

        # "Better" direction varies by dimension:
        # Higher is better: safety_score, substitute_count
        # Lower is better: side_effect_count, interaction_count, warning_count
        higher_is_better = dim in ("safety_score", "substitute_count")

        if higher_is_better:
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))

        winners[dim] = medicines[best_idx].get("name", f"Medicine {best_idx+1}")

    return {
        "scores": all_scores,
        "winners": winners,
        "medicines": medicines,
    }


# ─────────────────────────────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────────────────────────────

def generate_radar_chart(medicines: list, scores: list) -> plt.Figure:
    """
    Generate a matplotlib radar (spider) chart comparing medicines.

    Normalises all dimension scores to 0–1 scale (higher = better in all
    directions) so the chart is intuitive. Applies dark theme matching app.

    Args:
        medicines (list[dict]): Medicine data dicts (2 or 3).
        scores (list[dict]): Score dicts from compute_medicine_scores().

    Returns:
        matplotlib.figure.Figure: Dark-themed radar chart figure object.
                                   Pass to st.pyplot() in the UI.

    Example:
        >>> fig = generate_radar_chart([med1, med2], [scores1, scores2])
        >>> fig  # matplotlib Figure
    """
    # Labels for radar axes
    labels = [DIMENSION_LABELS[d] for d in COMPARISON_DIMENSIONS]
    num_dims = len(COMPARISON_DIMENSIONS)

    # Compute angles for each axis in a circle (equal spacing)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    # Close the loop by repeating the first angle at the end
    angles += angles[:1]

    # ── Normalise scores so higher = better on all axes ──────────────
    # For "lower is better" metrics, invert: normalised = 1 - (raw / max)
    # For "higher is better" metrics, normalise: raw / max
    max_values = {}
    for dim in COMPARISON_DIMENSIONS:
        max_val = max(s[dim] for s in scores)
        max_values[dim] = max_val if max_val > 0 else 1  # avoid division by zero

    def normalise_score(score_dict: dict) -> list:
        """Normalise a score dict to 0-1 values, inverting where needed."""
        normalised = []
        higher_is_better = {"safety_score", "substitute_count"}
        for dim in COMPARISON_DIMENSIONS:
            raw = score_dict[dim]
            if dim in higher_is_better:
                # Higher raw = higher normalised
                normalised.append(raw / max_values[dim])
            else:
                # Lower raw = higher normalised (invert for chart display)
                normalised.append(1.0 - (raw / max_values[dim]))
        return normalised

    # ── Create figure with dark background ───────────────────────────
    fig = plt.figure(figsize=(8, 7), facecolor=CHART_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(CHART_BG)

    # ── Style: dark theme grid ────────────────────────────────────────
    ax.spines["polar"].set_color("#1e3a5f")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.set_theta_offset(np.pi / 2)   # Start from top
    ax.set_theta_direction(-1)        # Clockwise

    # Draw faint grid circles
    ax.yaxis.set_tick_params(labelcolor="#475569")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=7, color="#475569")
    ax.grid(color="#1e3a5f", linewidth=0.8, linestyle="--")

    # ── Set axis labels ───────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, color="#94a3b8", fontweight="bold")

    # ── Plot each medicine ────────────────────────────────────────────
    for i, (med, score_dict) in enumerate(zip(medicines, scores)):
        norm_vals = normalise_score(score_dict)
        # Close the polygon by repeating first value
        values = norm_vals + norm_vals[:1]

        color = PROCESS_COLORS[i % len(PROCESS_COLORS)]
        med_name = med.get("name", f"Medicine {i+1}").title()

        # Draw filled polygon with semi-transparent fill
        ax.fill(angles, values, color=color, alpha=0.15)

        # Draw outline with glow-like thick line
        ax.plot(angles, values, color=color, linewidth=2.5,
                linestyle="solid", label=med_name, marker="o",
                markersize=6, markerfacecolor=color)

    # ── Legend ────────────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.15),
        fontsize=10,
        framealpha=0.0,
        labelcolor="#e2e8f0"
    )

    # ── Title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Medicine Comparison — Radar Chart",
        color="#00d4ff", fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# VERDICT GENERATION
# ─────────────────────────────────────────────────────────────────────

def generate_comparison_verdict(medicines: list, scores: list) -> str:
    """
    Generate a human-readable Markdown comparison verdict.

    Compares medicines on each dimension and generates natural language
    statements about which is better for what.

    Args:
        medicines (list[dict]): Medicine data dicts.
        scores (list[dict]): Computed score dicts.

    Returns:
        str: Markdown-formatted comparison verdict.

    Example:
        >>> verdict = generate_comparison_verdict([para, ibu], [s1, s2])
        >>> "fewer side effects" in verdict or "safer" in verdict
        True
    """
    names = [m.get("name", f"Medicine {i+1}").title() for i, m in enumerate(medicines)]
    lines = ["### 📊 Comparison Verdict\n"]

    # ── Safety comparison ─────────────────────────────────────────────
    safety_scores = [s["safety_score"] for s in scores]
    best_safety_idx = int(np.argmax(safety_scores))
    if len(set(safety_scores)) == 1:
        lines.append(f"🔒 **Safety:** {' and '.join(names)} have a **similar safety profile** "
                     f"based on contraindication count.")
    else:
        lines.append(f"🔒 **Safety:** **{names[best_safety_idx]}** appears safer with "
                     f"fewer absolute contraindications.")

    # ── Side effects comparison ───────────────────────────────────────
    se_counts = [s["side_effect_count"] for s in scores]
    best_se_idx = int(np.argmin(se_counts))  # fewer = better
    if len(set(se_counts)) == 1:
        lines.append(f"⚠️ **Side Effects:** Both medicines list a similar number of side effects.")
    else:
        lines.append(f"⚠️ **Side Effects:** **{names[best_se_idx]}** has fewer listed side effects "
                     f"({se_counts[best_se_idx]} vs. {max(se_counts)}).")

    # ── Interactions comparison ───────────────────────────────────────
    inter_counts = [s["interaction_count"] for s in scores]
    best_inter_idx = int(np.argmin(inter_counts))
    if len(set(inter_counts)) == 1:
        lines.append(f"🔗 **Drug Interactions:** Both have a similar interaction profile.")
    else:
        lines.append(f"🔗 **Drug Interactions:** **{names[best_inter_idx]}** has fewer documented "
                     f"interactions ({inter_counts[best_inter_idx]} vs. {max(inter_counts)}).")

    # ── Substitutes comparison ────────────────────────────────────────
    sub_counts = [s["substitute_count"] for s in scores]
    best_sub_idx = int(np.argmax(sub_counts))
    if len(set(sub_counts)) == 1:
        lines.append(f"🔄 **Alternatives:** Both medicines have a similar number of alternatives.")
    else:
        lines.append(f"🔄 **Alternatives:** **{names[best_sub_idx]}** has more available "
                     f"alternatives ({sub_counts[best_sub_idx]}).")

    # ── Overall recommendation ────────────────────────────────────────
    lines.append("\n---")
    lines.append(
        "⚠️ *This comparison is based on database information only. "
        "The 'better' choice always depends on your specific medical condition, "
        "history, and doctor's advice. Never self-medicate based solely on this comparison.*"
    )

    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────

def build_comparison_table(medicines: list) -> pd.DataFrame:
    """
    Build a pandas DataFrame for tabular side-by-side comparison display.

    Rows = comparison fields; Columns = medicine names.
    Long text is truncated at 120 characters for readability.

    Args:
        medicines (list[dict]): List of 2 or 3 medicine data dicts.

    Returns:
        pd.DataFrame: Comparison table with medicines as columns,
                      clinical fields as rows.

    Example:
        >>> df = build_comparison_table([para, ibu])
        >>> "Category" in df.index
        True
    """
    names = [m.get("name", f"Medicine {i+1}").title() for i, m in enumerate(medicines)]
    rows = {}

    for label, field in COMPARE_TABLE_FIELDS:
        row_values = []
        for med in medicines:
            val = med.get(field, "Not specified")

            # Convert list fields to comma-separated string
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val if v)
            else:
                val = str(val)

            # Truncate long text to keep table readable
            if len(val) > 120:
                val = val[:117] + "..."

            row_values.append(val)

        rows[label] = row_values

    # Create DataFrame with field labels as index, medicine names as columns
    df = pd.DataFrame(rows, index=names).T
    return df
