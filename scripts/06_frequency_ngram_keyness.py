#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_frequency_ngram_keyness.py

Purpose
-------
Generate lexical frequency and N-gram tables for the cleaned direct-message
and group-message corpora.

This script calculates:
- word counts before and after lexical cleaning
- token frequencies
- bigrams
- trigrams

The script uses the cleaned corpus tables produced by:

    scripts/03_clean_lexical.py

Inputs
------
Expected input files:

    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Each input file must contain:

    text_original
    text_clean_lexical

Outputs
-------
Aggregated tables are written to:

    outputs/public/tables/frequency_ngram_results.xlsx

Usage
-----
Run from the project root directory:

    python scripts/06_frequency_ngram_keyness.py --corpus direct
    python scripts/06_frequency_ngram_keyness.py --corpus group
    python scripts/06_frequency_ngram_keyness.py --corpus both

Optional:

    python scripts/06_frequency_ngram_keyness.py --corpus both --min-count 3
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CORPUS_CONFIG = {
    "direct": {
        "input_file": INPUT_DIR / "D_utterances_clean_lexical.csv",
        "prefix": "D",
        "label": "direct",
    },
    "group": {
        "input_file": INPUT_DIR / "G_utterances_clean_lexical.csv",
        "prefix": "G",
        "label": "group",
    },
}


# ---------------------------------------------------------------------
# Tokenization and counting helpers
# ---------------------------------------------------------------------

def simple_tokenize(text: str) -> list[str]:
    """
    Tokenize cleaned text using a simple whitespace-based tokenizer.

    This is appropriate after lexical cleaning because functional units
    such as PatName, KolName, Mention_KolName, Hashtag_PatName, and
    kein_Todo were intentionally preserved as single tokens.
    """

    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    return text.split()


def count_words_in_series(text_series: pd.Series) -> int:
    """
    Count whitespace-separated tokens in a pandas Series.
    """

    return (
        text_series
        .fillna("")
        .astype(str)
        .apply(lambda x: len(simple_tokenize(x)))
        .sum()
    )


def create_word_count_summary(df: pd.DataFrame, corpus_label: str) -> pd.DataFrame:
    """
    Create word count summary before and after lexical cleaning.
    """

    if "text_original" not in df.columns:
        raise ValueError("Input table must contain 'text_original'.")

    if "text_clean_lexical" not in df.columns:
        raise ValueError("Input table must contain 'text_clean_lexical'.")

    n_messages = len(df)

    original_word_count = count_words_in_series(df["text_original"])
    cleaned_word_count = count_words_in_series(df["text_clean_lexical"])

    original_mean = original_word_count / n_messages if n_messages > 0 else 0
    cleaned_mean = cleaned_word_count / n_messages if n_messages > 0 else 0

    token_difference = cleaned_word_count - original_word_count
    relative_difference_percent = (
    token_difference / original_word_count * 100
    if original_word_count > 0
    else 0
    )

    summary = pd.DataFrame(
        [
            {
                "corpus": corpus_label,
                "n_messages": n_messages,
                "word_count_original": original_word_count,
                "word_count_cleaned": cleaned_word_count,
                "mean_words_per_message_original": original_mean,
                "mean_words_per_message_cleaned": cleaned_mean,
                "token_difference_cleaned_minus_original": token_difference,
                "relative_difference_percent": relative_difference_percent,
            }
        ]
    )

    return summary


def create_token_frequency_table(
    df: pd.DataFrame,
    corpus_label: str,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Create token frequency table from cleaned text.
    """

    counter = Counter()

    for text in df["text_clean_lexical"].fillna("").astype(str):
        tokens = simple_tokenize(text)
        counter.update(tokens)

    records = []

    total_tokens = sum(counter.values())

    for token, count in counter.items():
        if count >= min_count:
            records.append(
                {
                    "corpus": corpus_label,
                    "token": token,
                    "count": count,
                    "relative_frequency_per_1000_tokens": (
                        count / total_tokens * 1000 if total_tokens > 0 else 0
                    ),
                }
            )

    result = pd.DataFrame(records)

    if len(result) > 0:
        result = result.sort_values(["count", "token"], ascending=[False, True])

    return result


def make_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """
    Create n-grams from a token list.
    """

    if len(tokens) < n:
        return []

    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def create_ngram_frequency_table(
    df: pd.DataFrame,
    corpus_label: str,
    n: int,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Create N-gram frequency table from cleaned text.
    """

    counter = Counter()

    for text in df["text_clean_lexical"].fillna("").astype(str):
        tokens = simple_tokenize(text)
        ngrams = make_ngrams(tokens, n)
        counter.update(ngrams)

    records = []
    total_ngrams = sum(counter.values())

    for ngram_tuple, count in counter.items():
        if count >= min_count:
            ngram = " ".join(ngram_tuple)
            records.append(
                {
                    "corpus": corpus_label,
                    "n": n,
                    "ngram": ngram,
                    "count": count,
                    "relative_frequency_per_1000_ngrams": (
                        count / total_ngrams * 1000 if total_ngrams > 0 else 0
                    ),
                }
            )

    result = pd.DataFrame(records)

    if len(result) > 0:
        result = result.sort_values(["count", "ngram"], ascending=[False, True])

    return result


# ---------------------------------------------------------------------
# Main corpus processing
# ---------------------------------------------------------------------

def analyze_corpus(corpus_name: str, min_count: int) -> dict[str, pd.DataFrame]:
    """
    Run word count, token frequency, bigram, and trigram analyses for one corpus.
    """

    config = CORPUS_CONFIG[corpus_name]

    print(f"Analyzing corpus: {corpus_name}")
    print(f"Input file: {config['input_file']}")

    if not config["input_file"].exists():
        raise FileNotFoundError(f"Input file not found: {config['input_file']}")

    df = pd.read_csv(config["input_file"])

    required_columns = ["text_original", "text_clean_lexical"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    word_counts = create_word_count_summary(df, config["label"])

    token_frequencies = create_token_frequency_table(
        df,
        config["label"],
        min_count=min_count,
    )

    bigrams = create_ngram_frequency_table(
        df,
        config["label"],
        n=2,
        min_count=min_count,
    )

    trigrams = create_ngram_frequency_table(
        df,
        config["label"],
        n=3,
        min_count=min_count,
    )

    return {
        "word_counts": word_counts,
        "token_frequencies": token_frequencies,
        "bigrams": bigrams,
        "trigrams": trigrams,
    }


def write_results_to_excel(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    """
    Write all results to a single Excel workbook with separate sheets.
    """

    output_file = OUTPUT_DIR / "frequency_ngram_results.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        word_count_tables = []
        token_frequency_tables = []
        bigram_tables = []
        trigram_tables = []

        for corpus_name, corpus_results in results.items():
            prefix = CORPUS_CONFIG[corpus_name]["prefix"]

            corpus_results["word_counts"].to_excel(
                writer,
                sheet_name=f"{prefix}_word_counts",
                index=False,
            )

            corpus_results["token_frequencies"].to_excel(
                writer,
                sheet_name=f"{prefix}_token_freq",
                index=False,
            )

            corpus_results["bigrams"].to_excel(
                writer,
                sheet_name=f"{prefix}_bigrams",
                index=False,
            )

            corpus_results["trigrams"].to_excel(
                writer,
                sheet_name=f"{prefix}_trigrams",
                index=False,
            )

            word_count_tables.append(corpus_results["word_counts"])
            token_frequency_tables.append(corpus_results["token_frequencies"])
            bigram_tables.append(corpus_results["bigrams"])
            trigram_tables.append(corpus_results["trigrams"])

        if word_count_tables:
            pd.concat(word_count_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_word_counts",
                index=False,
            )

        if token_frequency_tables:
            pd.concat(token_frequency_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_token_freq",
                index=False,
            )

        if bigram_tables:
            pd.concat(bigram_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_bigrams",
                index=False,
            )

        if trigram_tables:
            pd.concat(trigram_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_trigrams",
                index=False,
            )

    print(f"Saved results to: {output_file}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate word count, token frequency, and N-gram tables."
    )

    parser.add_argument(
        "--corpus",
        choices=["direct", "group", "both"],
        required=True,
        help="Choose which corpus to analyze: direct, group, or both.",
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum frequency threshold for tokens and N-grams.",
    )

    args = parser.parse_args()

    if args.corpus == "both":
        selected_corpora = ["direct", "group"]
    else:
        selected_corpora = [args.corpus]

    results = {}

    for corpus_name in selected_corpora:
        results[corpus_name] = analyze_corpus(
            corpus_name,
            min_count=args.min_count,
        )

    write_results_to_excel(results)


if __name__ == "__main__":
    main()