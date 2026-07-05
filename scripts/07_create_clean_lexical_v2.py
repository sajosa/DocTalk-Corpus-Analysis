#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_create_clean_lexical_v2.py

Purpose
-------
Create v2 cleaned lexical tables for targeted collocation and KWIC analyses.

This script applies a small number of additional rule-based normalizations to
the cleaned lexical corpus produced by:

    scripts/02_clean_lexical.py

v2 changes
----------
1. Normalize therapy group multiword expressions:

       MT Gruppe -> MT_Gruppe
       KT Gruppe -> KT_Gruppe
       GT Gruppe -> GT_Gruppe

2. Normalize explicit no-task status expressions:

       kein aktives Todo -> kein_Todo
       kein aktuelles Todo -> kein_Todo
       kein_aktives_Todo -> kein_Todo

3. Normalize selected sentence-initial pronouns/function words:

       Ich -> ich
       Du -> du
       Mir -> mir
       etc.

No separate content-versus-interaction views are generated in the final
pipeline. The v2 text column is used as the basis for targeted collocation
and KWIC analyses.

Inputs
------
Expected input files:

    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Outputs
-------
Confidential message-level outputs:

    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical_v2.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical_v2.csv
    outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical_v2.csv

Aggregated public validation output:

    outputs/public/tables/collocations_v2/marker_presence_check_clean_lexical_v2.xlsx

Usage
-----
Run from the project root directory:

    python scripts/07_create_clean_lexical_v2.py

Confidentiality
---------------
The cleaned v2 utterance tables still contain message-level text and are
therefore written to outputs/confidential/. The marker presence check contains
aggregated counts only and is written to outputs/public/.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"

DIRECT_IN = INPUT_DIR / "D_utterances_clean_lexical.csv"
GROUP_IN = INPUT_DIR / "G_utterances_clean_lexical.csv"

DIRECT_OUT = INPUT_DIR / "D_utterances_clean_lexical_v2.csv"
GROUP_OUT = INPUT_DIR / "G_utterances_clean_lexical_v2.csv"
COMBINED_OUT = INPUT_DIR / "utterances_for_collocation_clean_lexical_v2.csv"

MARKER_OUT = (
    PROJECT_DIR
    / "outputs"
    / "public"
    / "tables"
    / "collocations_v2"
    / "marker_presence_check_clean_lexical_v2.xlsx"
)

TEXT_COL = "text_clean_lexical"
V2_TEXT_COL = "text_clean_lexical_v2"


# -----------------------------------------------------------------------------
# Rule-based v2 normalization
# -----------------------------------------------------------------------------

THERAPY_GROUP_RULES = [
    (r"\bMT\s+Gruppe\b", "MT_Gruppe"),
    (r"\bKT\s+Gruppe\b", "KT_Gruppe"),
    (r"\bGT\s+Gruppe\b", "GT_Gruppe"),
]


TODO_STATUS_RULES = [
    # Hashtag artefact variants, e.g. original "#kein to do"
    (r"\bHashtag_kein\s+Todo\b", "kein_Todo"),
    (r"\bHashtag_keine\s+Todo\b", "kein_Todo"),

    # Underscore variants
    (r"\bkein_aktives_Todo\b", "kein_Todo"),
    (r"\bkein_aktives_ToDo\b", "kein_Todo"),
    (r"\bkein_aktuelles_Todo\b", "kein_Todo"),
    (r"\bkein_aktuelles_ToDo\b", "kein_Todo"),

    # Whitespace variants
    (r"\bkein\s+aktives\s+Todo\b", "kein_Todo"),
    (r"\bkein\s+aktives\s+ToDo\b", "kein_Todo"),
    (r"\bkein\s+aktuelles\s+Todo\b", "kein_Todo"),
    (r"\bkein\s+aktuelles\s+ToDo\b", "kein_Todo"),
]


CASE_NORMALIZATION_RULES = [
    # Selected sentence-initial pronouns/function words.
    # Do not lowercase the full text because protected markers must remain stable.
    (r"\bIch\b", "ich"),
    (r"\bDu\b", "du"),
    (r"\bMir\b", "mir"),
    (r"\bMich\b", "mich"),
    (r"\bDir\b", "dir"),
    (r"\bDich\b", "dich"),
    (r"\bWir\b", "wir"),
    (r"\bUns\b", "uns"),
    (r"\bSich\b", "sich"),
]


MARKERS_TO_CHECK = [
    "PatName",
    "Hashtag_PatName",
    "KolName",
    "Mention_KolName",
    "Übergabe",
    "WE",
    "Todo",
    "kein_Todo",
    "kein_aktives_Todo",
    "kein",
    "Rückmeldung",
    "Gruppe",
    "MT",
    "GT",
    "KT",
    "MT_Gruppe",
    "GT_Gruppe",
    "KT_Gruppe",
    "Raum",
    "ÖGD",
]


def safe_text(value: object) -> str:
    """
    Convert missing values to empty strings and all other values to strings.
    """
    if pd.isna(value):
        return ""

    return str(value)


def normalize_therapy_groups(text: str) -> str:
    """
    Normalize recurrent therapy group multiword expressions.
    """
    text = safe_text(text)

    for pattern, replacement in THERAPY_GROUP_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def normalize_todo_status(text: str) -> str:
    """
    Normalize explicit no-task status expressions.

    These variants all express absence of an active/pending task and are
    therefore mapped to the established marker kein_Todo.
    """
    text = safe_text(text)

    for pattern, replacement in TODO_STATUS_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def normalize_case_selected(text: str) -> str:
    """
    Normalize selected sentence-initial function words/pronouns without
    lowercasing protected markers such as PatName, KolName, WE, ÖGD,
    or MT_Gruppe.
    """
    text = safe_text(text)

    for pattern, replacement in CASE_NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)

    return text


def create_v2_text(text: str) -> str:
    """
    Apply all v2 normalization steps to one cleaned lexical text.
    """
    text = normalize_therapy_groups(text)
    text = normalize_todo_status(text)
    text = normalize_case_selected(text)

    return " ".join(text.split())


def add_v2_columns(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """
    Add v2 lexical text and corpus label to one utterance table.
    """
    df = df.copy()
    df["direction"] = corpus
    # "direction" is used throughout the pipeline as the Direct-vs-Group
    # corpus/modality label.

    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Required column not found: {TEXT_COL}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[V2_TEXT_COL] = df[TEXT_COL].apply(create_v2_text)

    return df


def count_marker(text: str, marker: str) -> int:
    """
    Count exact token-like occurrences of one marker.
    """
    text = safe_text(text)
    pattern = rf"(?<!\w){re.escape(marker)}(?!\w)"

    return len(re.findall(pattern, text))


def marker_presence_check(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Count selected markers in one text column.
    """
    rows = []

    for marker in MARKERS_TO_CHECK:
        counts = df[text_col].apply(lambda x: count_marker(x, marker))
        rows.append(
            {
                "text_column": text_col,
                "marker": marker,
                "messages_with_marker": int((counts > 0).sum()),
                "total_occurrences": int(counts.sum()),
            }
        )

    return pd.DataFrame(rows)


def read_input_table(path: Path, label: str) -> pd.DataFrame:
    """
    Read one cleaned lexical input table.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} input not found: {path}")

    return pd.read_csv(path)


def main() -> int:
    direct_df = read_input_table(DIRECT_IN, "Direct")
    group_df = read_input_table(GROUP_IN, "Group")

    print("Loaded Direct:", direct_df.shape)
    print("Loaded Group:", group_df.shape)

    direct_v2 = add_v2_columns(direct_df, "direct")
    group_v2 = add_v2_columns(group_df, "group")

    DIRECT_OUT.parent.mkdir(parents=True, exist_ok=True)

    direct_v2.to_csv(DIRECT_OUT, index=False, encoding="utf-8")
    group_v2.to_csv(GROUP_OUT, index=False, encoding="utf-8")

    combined = pd.concat([direct_v2, group_v2], ignore_index=True, sort=False)
    combined.to_csv(COMBINED_OUT, index=False, encoding="utf-8")

    marker_original = marker_presence_check(combined, TEXT_COL)
    marker_v2 = marker_presence_check(combined, V2_TEXT_COL)

    MARKER_OUT.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(MARKER_OUT, engine="openpyxl") as writer:
        marker_original.to_excel(
            writer,
            sheet_name="text_clean_lexical",
            index=False,
        )
        marker_v2.to_excel(
            writer,
            sheet_name="text_clean_lexical_v2",
            index=False,
        )

    print("Saved:")
    print(" -", DIRECT_OUT)
    print(" -", GROUP_OUT)
    print(" -", COMBINED_OUT)
    print(" -", MARKER_OUT)

    print("\nMarker check v2:")
    print(marker_v2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())