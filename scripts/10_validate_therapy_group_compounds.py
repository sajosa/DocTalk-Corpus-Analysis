#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_validate_therapy_group_compounds.py

Purpose
-------
Validate therapy group multiword expressions before v2 lexical normalization.

This script checks whether therapy-group expressions occur as separable
multiword patterns in the cleaned lexical corpus, for example:

    MT Gruppe / Gruppe MT
    GT Gruppe / Gruppe GT
    KT Gruppe / Gruppe KT

The validation supports the v2 normalization rule used in:

    scripts/07_create_clean_lexical_v2.py

Inputs
------
Expected input files:

    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Output
------
Confidential validation workbook:

    outputs/confidential/validation_tables/therapy_group_compound_validation.xlsx

Confidentiality
---------------
The output contains matched message text and KWIC-style context snippets.
It must therefore remain in outputs/confidential/.

Usage
-----
Run from the project root directory:

    python scripts/10_validate_therapy_group_compounds.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"

INPUT_FILES = {
    "direct": INPUT_DIR / "D_utterances_clean_lexical.csv",
    "group": INPUT_DIR / "G_utterances_clean_lexical.csv",
}

OUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "validation_tables"
    / "therapy_group_compound_validation.xlsx"
)

TEXT_COL = "text_clean_lexical"

PATTERNS = {
    "MT_Gruppe_forward": r"\bMT\s+Gruppe\b",
    "MT_Gruppe_reverse": r"\bGruppe\s+MT\b",
    "GT_Gruppe_forward": r"\bGT\s+Gruppe\b",
    "GT_Gruppe_reverse": r"\bGruppe\s+GT\b",
    "KT_Gruppe_forward": r"\bKT\s+Gruppe\b",
    "KT_Gruppe_reverse": r"\bGruppe\s+KT\b",
    "MT_Gruppe_hyphen": r"\bMT[-_]Gruppe\b",
    "GT_Gruppe_hyphen": r"\bGT[-_]Gruppe\b",
    "KT_Gruppe_hyphen": r"\bKT[-_]Gruppe\b",
}


def safe_text(value: object) -> str:
    """
    Convert missing values to empty strings and all other values to strings.
    """
    if pd.isna(value):
        return ""

    return str(value)


def count_pattern(text: str, pattern: str) -> int:
    """
    Count regex pattern occurrences in one text.
    """
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def extract_kwic(
    text: str,
    pattern: str,
    window_chars: int = 80,
    max_examples: int = 5,
) -> list[dict[str, str]]:
    """
    Extract character-window KWIC examples around regex matches.
    """
    examples = []

    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(0, match.start() - window_chars)
        end = min(len(text), match.end() + window_chars)

        examples.append(
            {
                "left_context": text[start:match.start()],
                "match": text[match.start():match.end()],
                "right_context": text[match.end():end],
            }
        )

        if len(examples) >= max_examples:
            break

    return examples


def read_input_tables() -> pd.DataFrame:
    """
    Read direct and group cleaned lexical tables and add a direction label.
    """
    frames = []

    for direction, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"{direction} input file not found: {path}")

        df = pd.read_csv(path)

        if TEXT_COL not in df.columns:
            raise ValueError(
                f"Column '{TEXT_COL}' not found in {path}. "
                f"Available columns: {df.columns.tolist()}"
            )

        df = df.copy()
        df["direction"] = direction
        frames.append(df)

        print(f"Loaded {direction}: {path}")
        print(f"Shape: {df.shape}")

    return pd.concat(frames, ignore_index=True, sort=False)


def detect_team_column(df: pd.DataFrame) -> str:
    """
    Identify the best available team/context column.
    """
    candidates = [
        "team_name_final",
        "Team Name",
        "team_name_harmonized",
        "team_name_normalized",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    df["team_context"] = "unknown"
    return "team_context"


def collect_matches(df: pd.DataFrame, team_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect matched messages and KWIC examples for all therapy-group patterns.
    """
    rows = []
    kwic_rows = []

    for pattern_name, pattern in PATTERNS.items():
        compound = (
            pattern_name
            .replace("_forward", "")
            .replace("_reverse", "")
            .replace("_hyphen", "")
        )

        for idx, row in df.iterrows():
            text = safe_text(row[TEXT_COL])
            n_occurrences = count_pattern(text, pattern)

            if n_occurrences == 0:
                continue

            base_record = {
                "pattern_name": pattern_name,
                "compound_target": compound,
                "direction": row.get("direction", "unknown"),
                "team_context": row.get(team_col, "unknown"),
                "conversation_id": row.get("conversation_id", ""),
                "message_id": row.get("id", idx),
            }

            rows.append(
                {
                    **base_record,
                    "occurrences_in_message": n_occurrences,
                    "text_clean_lexical": text,
                }
            )

            for example in extract_kwic(text, pattern):
                kwic_rows.append(
                    {
                        **base_record,
                        "left_context": example["left_context"],
                        "match": example["match"],
                        "right_context": example["right_context"],
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(kwic_rows)


def create_summary_tables(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create overall and by-direction/team summaries.
    """
    if matches.empty:
        summary = pd.DataFrame(
            columns=[
                "pattern_name",
                "compound_target",
                "direction",
                "team_context",
                "messages_with_pattern",
                "total_occurrences",
            ]
        )
        overall = pd.DataFrame(
            columns=[
                "compound_target",
                "pattern_name",
                "messages_with_pattern",
                "total_occurrences",
            ]
        )
        return overall, summary

    summary = (
        matches
        .groupby(
            ["pattern_name", "compound_target", "direction", "team_context"],
            dropna=False,
        )
        .agg(
            messages_with_pattern=("message_id", "nunique"),
            total_occurrences=("occurrences_in_message", "sum"),
        )
        .reset_index()
        .sort_values(
            by=["compound_target", "direction", "team_context", "total_occurrences"],
            ascending=[True, True, True, False],
        )
    )

    overall = (
        summary
        .groupby(["compound_target", "pattern_name"], dropna=False)
        .agg(
            messages_with_pattern=("messages_with_pattern", "sum"),
            total_occurrences=("total_occurrences", "sum"),
        )
        .reset_index()
        .sort_values(
            ["compound_target", "total_occurrences"],
            ascending=[True, False],
        )
    )

    return overall, summary


def main() -> int:
    df = read_input_tables()

    print("Combined shape:", df.shape)
    print("Columns:", df.columns.tolist())

    team_col = detect_team_column(df)

    matches, kwic = collect_matches(df, team_col)
    overall, summary = create_summary_tables(matches)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall_counts", index=False)
        summary.to_excel(writer, sheet_name="by_direction_team", index=False)
        matches.to_excel(writer, sheet_name="matched_messages", index=False)
        kwic.to_excel(writer, sheet_name="kwic_examples", index=False)

    print("Saved:", OUT_PATH)
    print("Overall counts:")
    print(overall)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())