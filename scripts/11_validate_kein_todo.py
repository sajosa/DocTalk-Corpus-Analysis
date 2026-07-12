#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_validate_kein_todo.py

Purpose
-------
Validate whether original no-task expressions such as "kein aktives Todo" or
"keine ToDos" are correctly normalized to the established marker "kein_Todo"
after v2 lexical cleaning.

This script supports the v2 normalization rule used in:

    scripts/07_create_clean_lexical_v2.py

Input
-----
Expected input file:

    outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical_v2.csv

Output
------
Confidential validation workbook:

    outputs/confidential/validation_tables/validate_original_kein_aktives_todo_v2.xlsx

Confidentiality
---------------
The output contains original and cleaned message-level text and must therefore
remain in outputs/confidential/.

Usage
-----
Run from the project root directory:

    python scripts/11_validate_kein_todo.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "cleaned_corpus_tables"
    / "utterances_for_collocation_clean_lexical_v2.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "validation_tables"
    / "validate_original_kein_aktives_todo_v2.xlsx"
)

PATTERN = re.compile(
    r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*to[\s-]?dos?\b|"
    r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*todos?\b",
    flags=re.IGNORECASE,
)

RESULT_COLUMNS = [
    "corpus",
    "id",
    "conversation_id",
    "text_original",
    "text_clean_lexical",
    "text_clean_lexical_v2",
    "contains_kein_Todo_after_v1_cleaning",
    "contains_kein_Todo_after_v2_cleaning",
]


def safe_text(value: object) -> str:
    """
    Convert missing values to empty strings and all other values to strings.
    """
    if pd.isna(value):
        return ""

    return str(value)


def validate_input_table(df: pd.DataFrame) -> None:
    """
    Validate that required columns are present.
    """
    required_columns = [
        "text_original",
        "text_clean_lexical",
        "text_clean_lexical_v2",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def collect_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect rows where original text contains no-task Todo expressions.
    """
    rows = []

    for _, row in df.iterrows():
        original = safe_text(row.get("text_original", ""))
        cleaned_v1 = safe_text(row.get("text_clean_lexical", ""))
        cleaned_v2 = safe_text(row.get("text_clean_lexical_v2", ""))

        if not PATTERN.search(original):
            continue

        rows.append(
            {
                "corpus": row.get("direction", ""),
                "id": row.get("id", ""),
                "conversation_id": row.get("conversation_id", ""),
                "text_original": original,
                "text_clean_lexical": cleaned_v1,
                "text_clean_lexical_v2": cleaned_v2,
                "contains_kein_Todo_after_v1_cleaning": "kein_Todo" in cleaned_v1,
                "contains_kein_Todo_after_v2_cleaning": "kein_Todo" in cleaned_v2,
            }
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def create_summary_table(
    result: pd.DataFrame,
    flag_column: str,
    label: str,
) -> pd.DataFrame:
    """
    Create a compact count table for one validation flag.
    """
    if result.empty:
        return pd.DataFrame(
            columns=[
                "cleaning_stage",
                "corpus",
                flag_column,
                "count",
            ]
        )

    summary = (
        result
        .groupby(["corpus", flag_column], dropna=False)
        .size()
        .reset_index(name="count")
    )

    summary.insert(0, "cleaning_stage", label)

    return summary


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    validate_input_table(df)

    print("Loaded:", INPUT_PATH)
    print("Shape:", df.shape)

    result = collect_matches(df)

    summary_v1 = create_summary_table(
        result,
        flag_column="contains_kein_Todo_after_v1_cleaning",
        label="v1_cleaning",
    )

    summary_v2 = create_summary_table(
        result,
        flag_column="contains_kein_Todo_after_v2_cleaning",
        label="v2_cleaning",
    )

    if result.empty:
        false_cases_v2 = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        false_cases_v2 = result[
            result["contains_kein_Todo_after_v2_cleaning"] == False  # noqa: E712
        ].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        result.to_excel(
            writer,
            sheet_name="matched_original_texts",
            index=False,
        )

        summary_v1.to_excel(
            writer,
            sheet_name="summary_v1",
            index=False,
        )

        summary_v2.to_excel(
            writer,
            sheet_name="summary_v2",
            index=False,
        )

        false_cases_v2.to_excel(
            writer,
            sheet_name="false_cases_after_v2",
            index=False,
        )

    print("\nMatched rows:", result.shape)

    print("\nV1 result:")
    if summary_v1.empty:
        print("No matching original no-task Todo expressions found.")
    else:
        print(summary_v1.to_string(index=False))

    print("\nV2 result:")
    if summary_v2.empty:
        print("No matching original no-task Todo expressions found.")
    else:
        print(summary_v2.to_string(index=False))

    print("\nFalse cases after v2 cleaning:")
    if false_cases_v2.empty:
        print("None")
    else:
        display_cols = [
            "corpus",
            "id",
            "conversation_id",
            "text_original",
            "text_clean_lexical_v2",
        ]
        print(false_cases_v2[display_cols].to_string(index=False))

    print(f"\nSaved validation table to: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())