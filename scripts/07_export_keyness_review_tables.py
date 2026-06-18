#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_export_keyness_review_tables.py

Purpose
-------
Create small review tables from the Direct-vs-Group keyness results.

This script reads:

    outputs/public/tables/keyness_direct_vs_group.xlsx

and writes:

    outputs/public/tables/keyness_review_top_items.xlsx

The review workbook contains separate sheets for Direct-typical and
Group-typical tokens/N-grams.

Usage
-----
Run from the project root directory:

    python scripts/07_export_keyness_review_tables.py

Optional:

    python scripts/07_export_keyness_review_tables.py --top-n 30
    python scripts/07_export_keyness_review_tables.py --min-total-count 10
"""

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "public"
    / "tables"
    / "keyness_direct_vs_group.xlsx"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "public"
    / "tables"
    / "keyness_review_top_items.xlsx"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SHEET_CONFIG = [
    {
        "input_sheet": "key_content_tokens",
        "item_column": "token",
        "direct_output": "content_tokens_direct",
        "group_output": "content_tokens_group",
    },
    {
        "input_sheet": "key_interaction_tokens",
        "item_column": "token",
        "direct_output": "interaction_tokens_direct",
        "group_output": "interaction_tokens_group",
    },
    {
        "input_sheet": "key_content_bigrams",
        "item_column": "ngram",
        "direct_output": "content_bigrams_direct",
        "group_output": "content_bigrams_group",
    },
    {
        "input_sheet": "key_interaction_bigrams",
        "item_column": "ngram",
        "direct_output": "interaction_bigrams_direct",
        "group_output": "interaction_bigrams_group",
    },
    {
        "input_sheet": "key_content_trigrams",
        "item_column": "ngram",
        "direct_output": "content_trigrams_direct",
        "group_output": "content_trigrams_group",
    },
    {
        "input_sheet": "key_interaction_trigrams",
        "item_column": "ngram",
        "direct_output": "interaction_trigrams_direct",
        "group_output": "interaction_trigrams_group",
    },
]


REVIEW_COLUMNS_BASE = [
    "view",
    "item_type",
    "direction",
    "direct_count",
    "group_count",
    "total_count",
    "direct_freq_per_1000",
    "group_freq_per_1000",
    "difference_per_1000_direct_minus_group",
    "log_ratio_direct_vs_group",
    "log_likelihood",
    "signed_log_likelihood",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def create_review_table(
    df: pd.DataFrame,
    item_column: str,
    direction: str,
    top_n: int,
    min_total_count: int,
) -> pd.DataFrame:
    """
    Create a smaller review table for one direction.

    For direction='direct':
        sort by signed_log_likelihood descending.

    For direction='group':
        sort by signed_log_likelihood ascending.
    """

    required_columns = {
        item_column,
        "direction",
        "total_count",
        "signed_log_likelihood",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in keyness table: {missing_columns}"
        )

    review_df = df.copy()

    review_df = review_df[
        (review_df["direction"] == direction)
        & (review_df["total_count"] >= min_total_count)
    ]

    if direction == "direct":
        review_df = review_df.sort_values(
            ["signed_log_likelihood", "total_count"],
            ascending=[False, False],
        )
    elif direction == "group":
        review_df = review_df.sort_values(
            ["signed_log_likelihood", "total_count"],
            ascending=[True, False],
        )
    else:
        raise ValueError("direction must be either 'direct' or 'group'.")

    review_df = review_df.head(top_n)

    review_columns = [item_column] + [
        col for col in REVIEW_COLUMNS_BASE if col in review_df.columns
    ]

    review_df = review_df[review_columns]

    return review_df


def read_keyness_sheet(input_file: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read one keyness sheet from the keyness workbook.
    """

    try:
        return pd.read_excel(input_file, sheet_name=sheet_name)
    except ValueError as error:
        raise ValueError(
            f"Could not read sheet '{sheet_name}' from {input_file}. "
            "Please check whether the keyness workbook was generated correctly."
        ) from error


def export_review_tables(
    input_file: Path,
    output_file: Path,
    top_n: int,
    min_total_count: int,
) -> None:
    """
    Export all review tables into one Excel workbook.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Please run first:\n"
            "python scripts/06_frequency_ngram_keyness.py --corpus both --min-count 3"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        overview_records = []

        for config in SHEET_CONFIG:
            input_sheet = config["input_sheet"]
            item_column = config["item_column"]

            print(f"Processing sheet: {input_sheet}")

            df = read_keyness_sheet(input_file, input_sheet)

            direct_review = create_review_table(
                df=df,
                item_column=item_column,
                direction="direct",
                top_n=top_n,
                min_total_count=min_total_count,
            )

            group_review = create_review_table(
                df=df,
                item_column=item_column,
                direction="group",
                top_n=top_n,
                min_total_count=min_total_count,
            )

            direct_sheet_name = config["direct_output"][:31]
            group_sheet_name = config["group_output"][:31]

            direct_review.to_excel(
                writer,
                sheet_name=direct_sheet_name,
                index=False,
            )

            group_review.to_excel(
                writer,
                sheet_name=group_sheet_name,
                index=False,
            )

            overview_records.append(
                {
                    "input_sheet": input_sheet,
                    "direct_review_sheet": direct_sheet_name,
                    "group_review_sheet": group_sheet_name,
                    "n_direct_rows": len(direct_review),
                    "n_group_rows": len(group_review),
                    "top_n": top_n,
                    "min_total_count": min_total_count,
                }
            )

        overview_df = pd.DataFrame(overview_records)

        overview_df.to_excel(
            writer,
            sheet_name="overview",
            index=False,
        )

    print(f"\nSaved review workbook to: {output_file}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create compact review tables from keyness results."
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top items per direction and table.",
    )

    parser.add_argument(
        "--min-total-count",
        type=int,
        default=10,
        help=(
            "Minimum total count across Direct and Group required for an item "
            "to appear in the review table."
        ),
    )

    args = parser.parse_args()

    export_review_tables(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        top_n=args.top_n,
        min_total_count=args.min_total_count,
    )


if __name__ == "__main__":
    main()