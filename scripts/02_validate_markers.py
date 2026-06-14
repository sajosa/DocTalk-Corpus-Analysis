#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_validate_markers.py

Purpose
-------
Generate validation tables before and after lexical cleaning.

This script checks the direct-message and/or group-message corpus tables for:
- anonymization placeholders: <...>
- hashtags: #...
- mentions: @...
- remaining functional marker characters after cleaning
- original-versus-cleaned message examples

The outputs may contain message-level text and are therefore written to:

    outputs/confidential/validation_tables/

Usage
-----
Run from the project root directory:

    python scripts/02_validate_markers.py --corpus direct
    python scripts/02_validate_markers.py --corpus group
    python scripts/02_validate_markers.py --corpus both

Optional:

    python scripts/02_validate_markers.py --corpus both --max-examples 200
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "dataframes"
CLEANED_INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "validation_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CORPUS_CONFIG = {
    "direct": {
        "raw_file": RAW_INPUT_DIR / "D_utterances_raw.csv",
        "cleaned_file": CLEANED_INPUT_DIR / "D_utterances_clean_lexical.csv",
        "prefix": "D",
    },
    "group": {
        "raw_file": RAW_INPUT_DIR / "G_utterances_raw.csv",
        "cleaned_file": CLEANED_INPUT_DIR / "G_utterances_clean_lexical.csv",
        "prefix": "G",
    },
}


# ---------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------

PLACEHOLDER_PATTERN = re.compile(r"<[^>\n]+>")
HASHTAG_PATTERN = re.compile(r"#[^\s#@]+")
MENTION_PATTERN = re.compile(r"@[^\s#@]+")
REMAINING_MARKER_PATTERN = re.compile(r"[<>#@]")

STANDARD_TOKENS = [
    "PatName",
    "KolName",
    "Mention_KolName",
    "Mention_All",
    "Hashtag_PatName",
    "Todo",
    "kein_Todo",
    "Datum",
    "Klinik",
    "Telefonnummer",
    "Link",
    "Link_intern",
]
# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def ensure_text_column(df: pd.DataFrame, preferred_columns: list[str]) -> str:
    """
    Return the first available text column from a list of preferred columns.
    """

    for col in preferred_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "None of the expected text columns were found. "
        f"Expected one of: {preferred_columns}. "
        f"Available columns: {list(df.columns)}"
    )


def extract_marker_counts(
    df: pd.DataFrame,
    text_col: str,
    pattern: re.Pattern,
    marker_type: str,
) -> pd.DataFrame:
    """
    Extract markers from a text column and return marker-level counts.

    The output includes:
    - marker type
    - marker string
    - total count
    - number of messages containing the marker
    - one example text
    """

    records = []

    for row_idx, text in df[text_col].fillna("").items():
        text = str(text)
        matches = pattern.findall(text)

        for marker in matches:
            records.append(
                {
                    "row_index": row_idx,
                    "marker_type": marker_type,
                    "marker": marker,
                    "text": text,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "marker_type",
                "marker",
                "count",
                "n_messages",
                "example_text",
            ]
        )

    marker_df = pd.DataFrame(records)

    summary = (
        marker_df
        .groupby(["marker_type", "marker"], as_index=False)
        .agg(
            count=("marker", "size"),
            n_messages=("row_index", "nunique"),
            example_text=("text", "first"),
        )
        .sort_values(["count", "marker"], ascending=[False, True])
    )

    return summary


def extract_rows_with_remaining_markers(
    df: pd.DataFrame,
    text_col: str,
) -> pd.DataFrame:
    """
    Return rows where cleaned text still contains <, >, #, or @.
    """

    working_df = df.copy()
    working_df[text_col] = working_df[text_col].fillna("").astype(str)

    mask = working_df[text_col].str.contains(REMAINING_MARKER_PATTERN, regex=True)

    cols_to_keep = [col for col in ["id", "conversation_id", "speaker", text_col] if col in working_df.columns]

    if not cols_to_keep:
        cols_to_keep = [text_col]

    result = working_df.loc[mask, cols_to_keep].copy()
    result["remaining_marker_chars"] = result[text_col].apply(
        lambda x: "".join(sorted(set(REMAINING_MARKER_PATTERN.findall(x))))
    )

    return result


def create_original_vs_cleaned_examples(
    df: pd.DataFrame,
    original_col: str,
    cleaned_col: str,
    max_examples: int,
) -> pd.DataFrame:
    """
    Create deterministic original-versus-cleaned examples.

    Only messages where original and cleaned text differ are included.
    """

    working_df = df.copy()

    working_df[original_col] = working_df[original_col].fillna("").astype(str)
    working_df[cleaned_col] = working_df[cleaned_col].fillna("").astype(str)

    changed = working_df[working_df[original_col] != working_df[cleaned_col]].copy()

    cols_to_keep = [
        col
        for col in [
            "id",
            "conversation_id",
            "speaker",
            original_col,
            cleaned_col,
        ]
        if col in changed.columns
    ]

    result = changed.loc[:, cols_to_keep].head(max_examples).copy()

    result["original_length"] = result[original_col].str.len()
    result["cleaned_length"] = result[cleaned_col].str.len()

    return result


def create_validation_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    raw_text_col: str,
    cleaned_text_col: str,
) -> pd.DataFrame:
    """
    Create a compact validation summary for one corpus.
    """

    raw_text = raw_df[raw_text_col].fillna("").astype(str)
    cleaned_text = cleaned_df[cleaned_text_col].fillna("").astype(str)

    summary = pd.DataFrame(
        [
            {
                "metric": "n_raw_messages",
                "value": len(raw_df),
            },
            {
                "metric": "n_cleaned_messages",
                "value": len(cleaned_df),
            },
            {
                "metric": "raw_placeholder_occurrences",
                "value": raw_text.apply(lambda x: len(PLACEHOLDER_PATTERN.findall(x))).sum(),
            },
            {
                "metric": "raw_hashtag_occurrences",
                "value": raw_text.apply(lambda x: len(HASHTAG_PATTERN.findall(x))).sum(),
            },
            {
                "metric": "raw_mention_occurrences",
                "value": raw_text.apply(lambda x: len(MENTION_PATTERN.findall(x))).sum(),
            },
            {
                "metric": "cleaned_remaining_placeholder_occurrences",
                "value": cleaned_text.apply(lambda x: len(PLACEHOLDER_PATTERN.findall(x))).sum(),
            },
            {
                "metric": "cleaned_remaining_hashtag_occurrences",
                "value": cleaned_text.apply(lambda x: len(HASHTAG_PATTERN.findall(x))).sum(),
            },
            {
                "metric": "cleaned_remaining_mention_occurrences",
                "value": cleaned_text.apply(lambda x: len(MENTION_PATTERN.findall(x))).sum(),
            },
            {
                "metric": "cleaned_rows_with_any_remaining_marker_char",
                "value": cleaned_text.str.contains(REMAINING_MARKER_PATTERN, regex=True).sum(),
            },
        ]
    )

    return summary

def count_standard_tokens(
    df: pd.DataFrame,
    text_col: str,
    standard_tokens: list[str],
) -> pd.DataFrame:
    """
    Count standardized tokens created by the cleaning pipeline.

    This helps validate whether relevant source markers were transformed
    into the expected target tokens.
    """

    text_series = df[text_col].fillna("").astype(str)

    records = []

    for token in standard_tokens:
        pattern = re.compile(rf"\b{re.escape(token)}\b")

        count = text_series.apply(
            lambda x: len(pattern.findall(x))
        ).sum()

        n_messages = text_series.apply(
            lambda x: bool(pattern.search(x))
        ).sum()

        example_text = ""

        examples = text_series[text_series.apply(lambda x: bool(pattern.search(x)))]
        if len(examples) > 0:
            example_text = examples.iloc[0]

        records.append(
            {
                "standard_token": token,
                "count": count,
                "n_messages": n_messages,
                "example_text": example_text,
            }
        )

    result = pd.DataFrame(records)
    result = result.sort_values("count", ascending=False)

    return result

# ---------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------

def validate_corpus(corpus_name: str, max_examples: int) -> None:
    """
    Run marker validation for one corpus.
    """

    config = CORPUS_CONFIG[corpus_name]
    prefix = config["prefix"]

    print(f"Validating corpus: {corpus_name}")

    if not config["raw_file"].exists():
        raise FileNotFoundError(f"Raw input file not found: {config['raw_file']}")

    if not config["cleaned_file"].exists():
        raise FileNotFoundError(f"Cleaned input file not found: {config['cleaned_file']}")

    raw_df = pd.read_csv(config["raw_file"])
    cleaned_df = pd.read_csv(config["cleaned_file"])

    raw_text_col = ensure_text_column(raw_df, ["text", "message", "utterance"])
    cleaned_text_col = ensure_text_column(cleaned_df, ["text_clean_lexical", "cleaned_text", "text"])

    if "text_original" in cleaned_df.columns:
        original_col = "text_original"
    elif "text" in cleaned_df.columns:
        original_col = "text"
    else:
        original_col = raw_text_col

    # Before cleaning
    placeholders_before = extract_marker_counts(
        raw_df,
        raw_text_col,
        PLACEHOLDER_PATTERN,
        "placeholder_before_cleaning",
    )

    hashtags_before = extract_marker_counts(
        raw_df,
        raw_text_col,
        HASHTAG_PATTERN,
        "hashtag_before_cleaning",
    )

    mentions_before = extract_marker_counts(
        raw_df,
        raw_text_col,
        MENTION_PATTERN,
        "mention_before_cleaning",
    )

    # After cleaning
    placeholders_after = extract_marker_counts(
        cleaned_df,
        cleaned_text_col,
        PLACEHOLDER_PATTERN,
        "placeholder_after_cleaning",
    )

    hashtags_after = extract_marker_counts(
        cleaned_df,
        cleaned_text_col,
        HASHTAG_PATTERN,
        "hashtag_after_cleaning",
    )

    mentions_after = extract_marker_counts(
        cleaned_df,
        cleaned_text_col,
        MENTION_PATTERN,
        "mention_after_cleaning",
    )

    rows_with_remaining_markers = extract_rows_with_remaining_markers(
        cleaned_df,
        cleaned_text_col,
    )

    original_vs_cleaned = create_original_vs_cleaned_examples(
        cleaned_df,
        original_col,
        cleaned_text_col,
        max_examples,
    )

    summary = create_validation_summary(
        raw_df,
        cleaned_df,
        raw_text_col,
        cleaned_text_col,
    )
    standard_token_counts = count_standard_tokens(
    cleaned_df,
    cleaned_text_col,
    STANDARD_TOKENS,
    )

    output_file = OUTPUT_DIR / f"{prefix}_cleaning_validation.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)

        standard_token_counts.to_excel(writer,sheet_name="standard_token_counts",
        index=False)

        placeholders_before.to_excel(writer, sheet_name="placeholders_before", index=False)
        hashtags_before.to_excel(writer, sheet_name="hashtags_before", index=False)
        mentions_before.to_excel(writer, sheet_name="mentions_before", index=False)

        placeholders_after.to_excel(writer, sheet_name="placeholders_after", index=False)
        hashtags_after.to_excel(writer, sheet_name="hashtags_after", index=False)
        mentions_after.to_excel(writer, sheet_name="mentions_after", index=False)

        rows_with_remaining_markers.to_excel(writer, sheet_name="remaining_marker_rows", index=False)
        original_vs_cleaned.to_excel(writer, sheet_name="original_vs_cleaned", index=False)

    print(f"Saved validation workbook to: {output_file}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate raw and cleaned corpus tables for remaining markers."
    )

    parser.add_argument(
        "--corpus",
        choices=["direct", "group", "both"],
        required=True,
        help="Choose which corpus to validate: direct, group, or both.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of original-versus-cleaned examples to export.",
    )

    args = parser.parse_args()

    if args.corpus == "both":
        validate_corpus("direct", args.max_examples)
        validate_corpus("group", args.max_examples)
    else:
        validate_corpus(args.corpus, args.max_examples)


if __name__ == "__main__":
    main()