###
# to start, run the following command in the terminal: 
# python scripts/03_clean_lexical.py --corpus direct
# or
# python scripts/03_clean_lexical.py --corpus group
# or
# python scripts/03_clean_lexical.py --corpus both
###

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_clean_lexical.py

Purpose
-------
Create corpus-specific cleaned versions of the direct-message and/or
group-message utterance tables for lexical analyses.

This script applies a rule-based cleaning pipeline to the message text.
The original exported utterance tables remain unchanged. The cleaned
outputs are intended for lexical frequency analyses, N-gram analyses,
and related corpus-linguistic analyses.

The cleaned corpus versions should not be used for temporal analyses,
conversation-structure analyses, emoji analyses, or sentiment analyses.

Inputs
------
Expected input files:

    outputs/confidential/dataframes/D_utterances_raw.csv
    outputs/confidential/dataframes/G_utterances_raw.csv

Each input table must contain a text column, usually named:

    text

Outputs
-------
Cleaned utterance tables are written to:

    outputs/confidential/cleaned_corpus_tables/

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
Run from the project root directory:

    python scripts/03_clean_lexical.py --corpus direct
    python scripts/03_clean_lexical.py --corpus group
    python scripts/03_clean_lexical.py --corpus both

Arguments
---------
--corpus:
    direct  Clean only the direct-message corpus.
    group   Clean only the group-message corpus.
    both    Clean both corpora sequentially.

Confidentiality
---------------
The outputs may still contain message-level text and must therefore be
treated as confidential. They are written to outputs/confidential/ and
should not be committed to GitHub.

Author
------
Sabine Sayegh-Jodehl

Project
-------
DocTalk chat corpus analysis / JMIR reproducibility pipeline
"""
import sys
from pathlib import Path
import argparse
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cleaning import clean_text_lexical

INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "dataframes"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CORPUS_CONFIG = {
    "direct": {
        "input_file": INPUT_DIR / "D_utterances_raw.csv",
        "output_file": OUTPUT_DIR / "D_utterances_clean_lexical.csv",
    },
    "group": {
        "input_file": INPUT_DIR / "G_utterances_raw.csv",
        "output_file": OUTPUT_DIR / "G_utterances_clean_lexical.csv",
    },
}


def clean_corpus(corpus_name: str):
    """
    Load one utterance table, apply lexical cleaning, and save the result.
    """

    config = CORPUS_CONFIG[corpus_name]

    print(f"Cleaning corpus: {corpus_name}")
    print(f"Input file: {config['input_file']}")

    if not config["input_file"].exists():
        raise FileNotFoundError(f"Input file not found: {config['input_file']}")

    df = pd.read_csv(config["input_file"])

    if "text" not in df.columns:
        raise ValueError(
            "Input table must contain a column named 'text'. "
            f"Available columns are: {list(df.columns)}"
        )

    df["text_original"] = df["text"]
    df["text_clean_lexical"] = df["text"].apply(clean_text_lexical)

    df.to_csv(config["output_file"], index=False)

    print(f"Saved cleaned corpus to: {config['output_file']}")


### main function to run the cleaning pipeline for the direct and/or group message corpora, depending on user input
def main():
    parser = argparse.ArgumentParser(
        description="Clean direct and/or group message corpora for lexical analyses."
    )

    parser.add_argument(
        "--corpus",
        choices=["direct", "group", "both"],
        required=True,
        help="Choose which corpus to clean: direct, group, or both.",
    )

    args = parser.parse_args()

    print(f"Selected corpus option: {args.corpus}")

    if args.corpus == "both":
        print("Cleaning both direct-message and group-message corpora...")
        clean_corpus("direct")
        clean_corpus("group")

    else:
        print(f"Cleaning {args.corpus}-message corpus...")
        clean_corpus(args.corpus)


if __name__ == "__main__":
    main()
