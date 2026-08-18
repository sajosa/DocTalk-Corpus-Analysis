#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_clean_lexical.py

Purpose
-------
Create corpus-specific cleaned versions of the direct-message and/or
group-message utterance tables for lexical analyses.

This script applies a rule-based cleaning pipeline to the message text.
The original exported utterance tables remain unchanged. The cleaned
outputs are intended for lexical frequency analyses, n-gram analyses,
keyness analyses, and related corpus-linguistic analyses.

The cleaned corpus versions should not be used for temporal analyses,
conversation-structure analyses, emoji analyses, or sentiment analyses.

Inputs
------
Default confidential input files:

    outputs/confidential/dataframes/D_utterances_raw.csv
    outputs/confidential/dataframes/G_utterances_raw.csv

Public synthetic demonstration input files:

    data/synthetic_sample/D_utterances_raw.csv
    data/synthetic_sample/G_utterances_raw.csv

Each input table must contain a message text column named:

    text

Outputs
-------
For the confidential corpus, cleaned utterance tables are written to:

    outputs/confidential/cleaned_corpus_tables/

For the public synthetic demonstration corpus, cleaned utterance tables
are written to:

    data/synthetic_sample/cleaned/

Generated files:

    D_utterances_clean_lexical.csv
    G_utterances_clean_lexical.csv

Main cleaning steps
-------------------
1. Standardize functional communication markers:
   - @-mentions
   - @all
   - hashtags

2. Remove export artefacts such as <enter>.

3. Harmonize anonymization placeholders:
   - colleague references -> KolName
   - patient-name references -> PatName
   - informative placeholders -> standardized tokens

4. Preserve semantically meaningful negations:
   - "kein ToDo" variants -> kein_Todo

5. Remove residual punctuation and normalize whitespace.

Usage
-----
Run from the project root directory.

Confidential corpus:

    python scripts/02_clean_lexical.py --corpus direct
    python scripts/02_clean_lexical.py --corpus group
    python scripts/02_clean_lexical.py --corpus both

Public synthetic demonstration corpus:

    python scripts/02_clean_lexical.py --corpus direct --synthetic
    python scripts/02_clean_lexical.py --corpus group --synthetic
    python scripts/02_clean_lexical.py --corpus both --synthetic

Confidentiality
---------------
Outputs generated from the confidential corpus contain message-level text
and must therefore be treated as confidential. They are written to
outputs/confidential/ and should not be committed to a public repository.

Outputs generated with --synthetic are based exclusively on the fully
synthetic demonstration corpus and are written to data/synthetic_sample/.

Project
-------
DocTalk chat corpus analysis / reproducibility pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cleaning import clean_text_lexical  # noqa: E402


CONFIDENTIAL_INPUT_DIR = (
    PROJECT_DIR / "outputs" / "confidential" / "dataframes"
)
CONFIDENTIAL_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"
)

SYNTHETIC_INPUT_DIR = (
    PROJECT_DIR / "data" / "synthetic_sample"
)
SYNTHETIC_OUTPUT_DIR = (
    PROJECT_DIR / "data" / "synthetic_sample" / "cleaned"
)


def get_corpus_config(
    synthetic: bool,
) -> dict[str, dict[str, Path]]:
    """
    Return input and output paths for the selected corpus source.

    If synthetic is True, use the public synthetic demonstration corpus.
    Otherwise, use the confidential corpus.
    """
    if synthetic:
        input_dir = SYNTHETIC_INPUT_DIR
        output_dir = SYNTHETIC_OUTPUT_DIR
    else:
        input_dir = CONFIDENTIAL_INPUT_DIR
        output_dir = CONFIDENTIAL_OUTPUT_DIR

    return {
        "direct": {
            "input_file": input_dir / "D_utterances_raw.csv",
            "output_file": output_dir / "D_utterances_clean_lexical.csv",
        },
        "group": {
            "input_file": input_dir / "G_utterances_raw.csv",
            "output_file": output_dir / "G_utterances_clean_lexical.csv",
        },
    }


def clean_corpus(
    corpus_name: str,
    synthetic: bool = False,
) -> dict[str, object]:
    """
    Load one utterance table, apply lexical cleaning, and save the result.

    The original message text is preserved in text_original. The cleaned text
    is stored in text_clean_lexical.
    """
    config = get_corpus_config(synthetic)[corpus_name]

    input_file = config["input_file"]
    output_file = config["output_file"]

    print(f"Cleaning corpus: {corpus_name}")
    print(
        "Data source: "
        + ("synthetic demonstration corpus" if synthetic else "confidential corpus")
    )
    print(f"Input file: {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    if "text" not in df.columns:
        raise ValueError(
            "Input table must contain a column named 'text'. "
            f"Available columns are: {list(df.columns)}"
        )

    df["text_original"] = df["text"]

    df["text_clean_lexical"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .apply(clean_text_lexical)
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "corpus": corpus_name,
        "synthetic": synthetic,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "n_rows": int(len(df)),
        "n_empty_cleaned_texts": int(
            (
                df["text_clean_lexical"]
                .astype(str)
                .str.strip()
                == ""
            ).sum()
        ),
    }

    print(f"Saved cleaned corpus to: {output_file}")
    print(f"Rows: {summary['n_rows']}")
    print(
        "Empty cleaned texts: "
        f"{summary['n_empty_cleaned_texts']}"
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean direct and/or group message corpora "
            "for lexical analyses."
        )
    )

    parser.add_argument(
        "--corpus",
        choices=["direct", "group", "both"],
        required=True,
        help=(
            "Choose which corpus to clean: "
            "direct, group, or both."
        ),
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use the public synthetic demonstration corpus "
            "instead of the confidential corpus."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Selected corpus option: {args.corpus}")
    print(
        "Selected data source: "
        + (
            "synthetic demonstration corpus"
            if args.synthetic
            else "confidential corpus"
        )
    )

    if args.corpus == "both":
        print(
            "Cleaning both direct-message "
            "and group-message corpora..."
        )

        clean_corpus(
            "direct",
            synthetic=args.synthetic,
        )

        clean_corpus(
            "group",
            synthetic=args.synthetic,
        )

    else:
        print(
            f"Cleaning {args.corpus}-message corpus..."
        )

        clean_corpus(
            args.corpus,
            synthetic=args.synthetic,
        )

    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())