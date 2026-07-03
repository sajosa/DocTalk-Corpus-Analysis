#!/usr/bin/env python3
"""
Sort review_top_collocates for manual interpretation.

Input:
    outputs/results/collocations/targeted_collocations_kwic_results_clean_lexical.xlsx

Output:
    outputs/results/collocations/review_top_collocates_sorted_for_manual_review.xlsx

Usage from project root:
    python scripts/09c_sort_collocation_review.py
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path.cwd()

IN_PATH = PROJECT_DIR / "outputs/results/collocations/targeted_collocations_kwic_results_clean_lexical.xlsx"
OUT_PATH = PROJECT_DIR / "outputs/results/collocations/review_top_collocates_sorted_for_manual_review.xlsx"

SHEET_NAME = "review_top_collocates"


def main():
    df = pd.read_excel(IN_PATH, sheet_name=SHEET_NAME)

    print("Loaded:", IN_PATH)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Put group before direct because your group-specific handover/status patterns
    # are likely central for the JMIR result table.
    direction_order = {
        "group": 0,
        "direct": 1,
    }

    df["direction_order"] = df["direction"].map(direction_order).fillna(99)

    sort_cols = [
        "anchor",
        "direction_order",
        "view",
        "frequency",
        "association_score",
        "ll_g2",
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
        .drop(columns=["direction_order"])
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_excel(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Shape:", df_sorted.shape)


if __name__ == "__main__":
    main()