#!/usr/bin/env python3
"""
Sort review_top_collocates for manual interpretation V2.

Input:
    outputs/confidential/review_files/collocations_v2/targeted_collocations_kwic_results_clean_lexical_v2.xlsx

Output:
    outputs/confidential/review_files/collocations_v2/review_top_collocates_sorted_for_manual_review_v2.xlsx

Usage from project root:
    python scripts/09_sort_collocation_review_v2.py
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path.cwd()

IN_PATH = PROJECT_DIR / "outputs/confidential/review_files/collocations_v2/targeted_collocations_kwic_results_clean_lexical_v2.xlsx"
OUT_PATH = PROJECT_DIR / "outputs/confidential/review_files/collocations_v2/review_top_collocates_sorted_for_manual_review_v2.xlsx"

SHEET_NAME = "review_top_collocates"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    df = pd.read_excel(IN_PATH, sheet_name=SHEET_NAME)

    print("Loaded:", IN_PATH)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Put group before direct because group-specific handover/status patterns
    # and documentation patterns, while preserving direct-message patterns.
    direction_order = {
        "group": 0,
        "direct": 1,
    }

    df["direction_order"] = df["direction"].map(direction_order).fillna(99)

    # V2 profile-oriented anchor order.
    # This makes the manual review easier by grouping the central result profiles:
    # 1. handover/task-status/no-task marking
    # 2. therapy group documentation incl. MT_Gruppe/KT_Gruppe/GT_Gruppe
    # 3. patient/colleague coordination and addressivity
    # 4. endoscopy-specific procedural/resource coordination
    anchor_profile_order = {
        # Handover / task-status / no-task marking
        "Hashtag_PatName": 0,
        "Übergabe": 1,
        "WE": 2,
        "kein_Todo": 3,
        "Todo": 4,
        "kein": 5,

        # Therapy group documentation and attendance
        "Rückmeldung": 10,
        "Gruppe": 11,
        "MT_Gruppe": 12,
        "KT_Gruppe": 13,
        "GT_Gruppe": 14,
        "MT": 15,
        "KT": 16,
        "GT": 17,

        # Bilateral case coordination / addressivity
        "PatName": 20,
        "KolName": 21,
        "Mention_KolName": 22,

        # Endoscopy-specific procedural/resource coordination
        "Raum": 30,
        "ÖGD": 31,
    }

    df["anchor_profile_order"] = df["anchor"].map(anchor_profile_order).fillna(99)

    sort_cols = [
        "anchor_profile_order",
        "anchor",
        "direction_order",
        "view",
        "frequency",
        "ll_g2",
        "association_score",
    ]

    existing_sort_cols = [col for col in sort_cols if col in df.columns]

    ascending = []
    for col in existing_sort_cols:
        if col in ["frequency", "association_score", "ll_g2"]:
            ascending.append(False)
        else:
            ascending.append(True)

    df_sorted = (
        df.sort_values(
            by=existing_sort_cols,
            ascending=ascending,
        )
        .drop(columns=["direction_order", "anchor_profile_order"])
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_excel(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Shape:", df_sorted.shape)

    print("\nTop 30 sorted review rows:")
    display_cols = [
        "anchor",
        "collocate",
        "direction",
        "view",
        "frequency",
        "association_score",
        "ll_g2",
        "possible_category",
        "interpretation_note",
    ]
    display_cols = [c for c in display_cols if c in df_sorted.columns]

    print(df_sorted[display_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()