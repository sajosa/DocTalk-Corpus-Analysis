#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_validate_unlinked_kein.py

Purpose
-------
Create a compact validation table for occurrences of the token "kein"
in the cleaned lexical corpus.

This helps decide whether "kein" is:
- part of a missed kein_Todo pattern
- a meaningful negation that should be retained
- a low-information token that can be moved to stopwords

Output
------
outputs/confidential/validation_tables/validate_unlinked_kein.xlsx

Usage
-----
Run from project root:

    python scripts/08_validate_unlinked_kein.py
"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "direct": PROJECT_DIR
    / "outputs"
    / "confidential"
    / "cleaned_corpus_tables"
    / "D_utterances_clean_lexical.csv",

    "group": PROJECT_DIR
    / "outputs"
    / "confidential"
    / "cleaned_corpus_tables"
    / "G_utterances_clean_lexical.csv",
}

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "validation_tables"
)

OUTPUT_FILE = OUTPUT_DIR / "validate_unlinked_kein.xlsx"

TARGET = "kein"
WINDOW = 6


def get_token_context(tokens: list[str], position: int, window: int) -> dict:
    """
    Return left and right token context around a target position.
    """

    left_start = max(position - window, 0)
    right_end = min(position + window + 1, len(tokens))

    left_tokens = tokens[left_start:position]
    right_tokens = tokens[position + 1:right_end]

    return {
        "left_context": " ".join(left_tokens),
        "target": tokens[position],
        "right_context": " ".join(right_tokens),
        "context_full": " ".join(tokens[left_start:right_end]),
        "token_before": tokens[position - 1] if position > 0 else "",
        "token_after": tokens[position + 1] if position + 1 < len(tokens) else "",
    }


def classify_context(token_before: str, token_after: str) -> str:
    """
    Provide a simple pre-classification for manual review.

    This is only a helper, not a final decision.
    """

    token_before_lower = token_before.lower()
    token_after_lower = token_after.lower()

    todo_like_after = {
        "todo",
        "todos",
        "to",
        "todo's",
        "todods",
        "aufgabe",
        "aufgaben",
        "bedarf",
        "handlungsbedarf",
    }

    modifier_after = {
        "akutes",
        "akute",
        "akuten",
        "aktives",
        "aktive",
        "aktiven",
        "offenes",
        "offene",
        "offenen",
        "neues",
        "neue",
        "neuen",
        "weiteres",
        "weitere",
        "weiteren",
    }

    if token_before == "Hashtag_PatName":
        return "check_after_patient_hashtag"

    if token_after_lower in todo_like_after:
        return "possible_missed_kein_Todo"

    if token_after_lower in modifier_after:
        return "possible_missed_kein_Todo_modifier"

    if token_after_lower == "Todo" or token_after == "Todo":
        return "possible_missed_kein_Todo"

    if token_after_lower == "mehr":
        return "negation_keine_mehr"

    if token_after_lower in {"ahnung", "problem", "probleme", "zeit"}:
        return "meaningful_negation"

    return "manual_review"


def collect_kein_rows(corpus: str, path: Path) -> pd.DataFrame:
    """
    Collect all rows where the cleaned text contains the exact token 'kein'.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required_columns = ["text_original", "text_clean_lexical"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    records = []

    for row_index, row in df.iterrows():
        cleaned = str(row.get("text_clean_lexical", ""))
        original = str(row.get("text_original", ""))

        tokens = cleaned.split()

        for position, token in enumerate(tokens):
            if token.lower() != TARGET:
                continue

            context = get_token_context(tokens, position, WINDOW)

            preclassification = classify_context(
                token_before=context["token_before"],
                token_after=context["token_after"],
            )

            record = {
                "corpus": corpus,
                "row_index": row_index,
                "conversation_id": row.get("conversation_id", ""),
                "token_before": context["token_before"],
                "target": context["target"],
                "token_after": context["token_after"],
                "left_context": context["left_context"],
                "right_context": context["right_context"],
                "context_full": context["context_full"],
                "preclassification": preclassification,
                "manual_decision": "",
                "manual_note": "",
                "cleaned_text": cleaned,
                "original_text": original,
            }

            records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_tables = []

    for corpus, path in INPUT_FILES.items():
        print(f"Checking {corpus}: {path}")
        result_tables.append(collect_kein_rows(corpus, path))

    result = pd.concat(result_tables, ignore_index=True)

    if result.empty:
        result = pd.DataFrame(
            [
                {
                    "message": "No exact token 'kein' found in cleaned lexical corpus."
                }
            ]
        )

    # Summary by token before/after
    if "token_before" in result.columns and "token_after" in result.columns:
        summary_before_after = (
            result
            .groupby(["token_before", "token_after"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        summary_preclassification = (
            result
            .groupby("preclassification", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        summary_before_after = pd.DataFrame()
        summary_preclassification = pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        result.to_excel(
            writer,
            sheet_name="kein_contexts",
            index=False,
        )

        summary_before_after.to_excel(
            writer,
            sheet_name="summary_before_after",
            index=False,
        )

        summary_preclassification.to_excel(
            writer,
            sheet_name="summary_preclass",
            index=False,
        )

    print(f"\nSaved validation table to: {OUTPUT_FILE}")
    print(f"Number of rows: {len(result)}")


if __name__ == "__main__":
    main()