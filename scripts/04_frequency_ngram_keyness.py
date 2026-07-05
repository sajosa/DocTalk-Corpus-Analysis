#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_frequency_ngram_keyness.py

Purpose
-------
Generate lexical frequency, N-gram, and Direct-vs-Group keyness tables for
the cleaned direct-message and group-message corpora.

This script calculates:
- word counts before and after lexical cleaning
- token frequencies based on all cleaned tokens
- bigram frequencies based on all cleaned tokens
- trigram frequencies based on all cleaned tokens
- Direct-vs-Group keyness for tokens, bigrams, and trigrams

Important methodological note
-----------------------------
N-grams are created from the full cleaned token sequence. No separate
content-versus-interaction views are generated in the final pipeline.

Public frequency and N-gram tables apply the user-defined --min-count
threshold. Direct-vs-Group keyness is computed from unthresholded internal
frequency tables and then filtered by the user-defined minimum total count.
This avoids treating low-frequency items in one corpus as zero when they are
present but below the public reporting threshold.

The script uses the cleaned corpus tables produced by:

    scripts/02_clean_lexical.py

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
Aggregated public tables are written to:

    outputs/public/tables/frequency_ngram_results.xlsx
    outputs/public/tables/keyness_direct_vs_group.xlsx

Usage
-----
Run from the project root directory:

    python scripts/04_frequency_ngram_keyness.py --corpus direct
    python scripts/04_frequency_ngram_keyness.py --corpus group
    python scripts/04_frequency_ngram_keyness.py --corpus both

Optional:

    python scripts/04_frequency_ngram_keyness.py --corpus both --min-count 3

Confidentiality
---------------
This script writes aggregated frequency and keyness tables only. No original
message-level text is exported. The default minimum count is 3 to reduce the
risk of exporting rare text-adjacent N-grams to public output tables.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "outputs" / "confidential" / "cleaned_corpus_tables"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public" / "tables"

FREQUENCY_OUTPUT_FILE = OUTPUT_DIR / "frequency_ngram_results.xlsx"
KEYNESS_OUTPUT_FILE = OUTPUT_DIR / "keyness_direct_vs_group.xlsx"


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
# Tokenization helpers
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


# ---------------------------------------------------------------------
# Word counts
# ---------------------------------------------------------------------

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

    return pd.DataFrame(
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


# ---------------------------------------------------------------------
# Frequency and N-gram helpers
# ---------------------------------------------------------------------

def create_token_frequency_table(
    df: pd.DataFrame,
    corpus_label: str,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Create token frequency table from cleaned text.
    """
    counter: Counter[str] = Counter()

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
                    "view": "all",
                    "token": token,
                    "count": count,
                    "relative_frequency_per_1000_tokens": (
                        count / total_tokens * 1000 if total_tokens > 0 else 0
                    ),
                    "total_tokens_in_view": total_tokens,
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
    counter: Counter[tuple[str, ...]] = Counter()

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
                    "view": "all",
                    "n": n,
                    "ngram": ngram,
                    "count": count,
                    "relative_frequency_per_1000_ngrams": (
                        count / total_ngrams * 1000 if total_ngrams > 0 else 0
                    ),
                    "total_ngrams_in_view": total_ngrams,
                }
            )

    result = pd.DataFrame(records)

    if len(result) > 0:
        result = result.sort_values(["count", "ngram"], ascending=[False, True])

    return result


# ---------------------------------------------------------------------
# Keyness helpers
# ---------------------------------------------------------------------

def _get_total_from_frequency_table(df: pd.DataFrame) -> int:
    """
    Extract the total number of tokens or N-grams in the analytical view.
    """
    if df.empty:
        return 0

    if "total_tokens_in_view" in df.columns:
        return int(df["total_tokens_in_view"].dropna().iloc[0])

    if "total_ngrams_in_view" in df.columns:
        return int(df["total_ngrams_in_view"].dropna().iloc[0])

    raise ValueError(
        "Frequency table must contain either 'total_tokens_in_view' "
        "or 'total_ngrams_in_view'."
    )


def _safe_log2(value: float) -> float:
    """
    Compute log2 safely.
    """
    if value <= 0:
        return 0.0

    return math.log2(value)


def compute_keyness_table(
    direct_df: pd.DataFrame,
    group_df: pd.DataFrame,
    item_column: str,
    item_type: str,
    min_total_count: int = 3,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """
    Compute Direct-vs-Group keyness for tokens or N-grams.

    Metrics:
    - relative frequencies per 1000
    - absolute difference per 1000
    - smoothed log ratio
    - smoothed odds ratio
    - log-likelihood

    Positive log_ratio / signed_log_likelihood indicates higher relative
    frequency in direct messages. Negative values indicate higher relative
    frequency in group messages.
    """
    if direct_df.empty and group_df.empty:
        return pd.DataFrame()

    direct_total = _get_total_from_frequency_table(direct_df)
    group_total = _get_total_from_frequency_table(group_df)

    if direct_total == 0 or group_total == 0:
        return pd.DataFrame()

    direct_counts = (
        direct_df.set_index(item_column)["count"].to_dict()
        if not direct_df.empty
        else {}
    )

    group_counts = (
        group_df.set_index(item_column)["count"].to_dict()
        if not group_df.empty
        else {}
    )

    all_items = sorted(set(direct_counts.keys()) | set(group_counts.keys()))

    records = []

    for item in all_items:
        direct_count = int(direct_counts.get(item, 0))
        group_count = int(group_counts.get(item, 0))
        total_count = direct_count + group_count

        if total_count < min_total_count:
            continue

        direct_freq_per_1000 = direct_count / direct_total * 1000
        group_freq_per_1000 = group_count / group_total * 1000

        difference_per_1000 = direct_freq_per_1000 - group_freq_per_1000

        direct_prop_smoothed = (direct_count + smoothing) / (
            direct_total + smoothing
        )
        group_prop_smoothed = (group_count + smoothing) / (
            group_total + smoothing
        )

        log_ratio = _safe_log2(direct_prop_smoothed / group_prop_smoothed)

        direct_non_count = max(direct_total - direct_count, 0)
        group_non_count = max(group_total - group_count, 0)

        direct_odds = (direct_count + smoothing) / (
            direct_non_count + smoothing
        )
        group_odds = (group_count + smoothing) / (
            group_non_count + smoothing
        )

        odds_ratio_smoothed = direct_odds / group_odds if group_odds > 0 else 0

        observed_total = direct_count + group_count
        corpus_total = direct_total + group_total

        expected_direct = direct_total * observed_total / corpus_total
        expected_group = group_total * observed_total / corpus_total

        log_likelihood = 0.0

        if direct_count > 0 and expected_direct > 0:
            log_likelihood += direct_count * math.log(
                direct_count / expected_direct
            )

        if group_count > 0 and expected_group > 0:
            log_likelihood += group_count * math.log(
                group_count / expected_group
            )

        log_likelihood *= 2

        if difference_per_1000 > 0:
            direction = "direct"
            signed_log_likelihood = log_likelihood
        elif difference_per_1000 < 0:
            direction = "group"
            signed_log_likelihood = -log_likelihood
        else:
            direction = "balanced"
            signed_log_likelihood = 0.0

        records.append(
            {
                "view": "all",
                "item_type": item_type,
                item_column: item,
                "direction": direction,
                "direct_count": direct_count,
                "group_count": group_count,
                "total_count": total_count,
                "direct_total": direct_total,
                "group_total": group_total,
                "direct_freq_per_1000": direct_freq_per_1000,
                "group_freq_per_1000": group_freq_per_1000,
                "difference_per_1000_direct_minus_group": difference_per_1000,
                "log_ratio_direct_vs_group": log_ratio,
                "abs_log_ratio": abs(log_ratio),
                "odds_ratio_smoothed": odds_ratio_smoothed,
                "log_likelihood": log_likelihood,
                "signed_log_likelihood": signed_log_likelihood,
            }
        )

    result = pd.DataFrame(records)

    if len(result) > 0:
        result = result.sort_values(
            ["log_likelihood", "abs_log_ratio", "total_count"],
            ascending=[False, False, False],
        )

    return result


def create_keyness_results(
    results: dict[str, dict[str, pd.DataFrame]],
    min_total_count: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Create Direct-vs-Group keyness tables for all-token analytical views.

    Keyness is computed for:
    - all tokens
    - all bigrams
    - all trigrams
    """
    if "direct" not in results or "group" not in results:
        print(
            "Skipping keyness analysis because both direct and group "
            "corpora are required."
        )
        return {}

    direct_results = results["direct"]
    group_results = results["group"]

    specs = [
        {
            "name": "key_all_tokens",
            "direct_key": "_keyness_token_frequencies_all",
            "group_key": "_keyness_token_frequencies_all",
            "item_column": "token",
            "item_type": "token",
        },
        {
            "name": "key_all_bigrams",
            "direct_key": "_keyness_bigrams_all",
            "group_key": "_keyness_bigrams_all",
            "item_column": "ngram",
            "item_type": "bigram",
        },
        {
            "name": "key_all_trigrams",
            "direct_key": "_keyness_trigrams_all",
            "group_key": "_keyness_trigrams_all",
            "item_column": "ngram",
            "item_type": "trigram",
        },
    ]

    keyness_results = {}

    for spec in specs:
        keyness_results[spec["name"]] = compute_keyness_table(
            direct_df=direct_results[spec["direct_key"]],
            group_df=group_results[spec["group_key"]],
            item_column=spec["item_column"],
            item_type=spec["item_type"],
            min_total_count=min_total_count,
        )

    return keyness_results


# ---------------------------------------------------------------------
# Main corpus processing
# ---------------------------------------------------------------------

def analyze_corpus(
    corpus_name: str,
    min_count: int,
) -> dict[str, pd.DataFrame]:
    """
    Run word count, token frequency, bigram, and trigram analyses for one corpus.

    Public frequency and N-gram outputs use min_count. Internal keyness source
    tables use min_count=1 to avoid biased zero counts in Direct-vs-Group
    comparisons.
    """
    config = CORPUS_CONFIG[corpus_name]

    print(f"\nAnalyzing corpus: {corpus_name}")
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

    token_frequencies_all = create_token_frequency_table(
        df,
        config["label"],
        min_count=min_count,
    )

    bigrams_all = create_ngram_frequency_table(
        df,
        config["label"],
        n=2,
        min_count=min_count,
    )

    trigrams_all = create_ngram_frequency_table(
        df,
        config["label"],
        n=3,
        min_count=min_count,
    )

    keyness_token_frequencies_all = create_token_frequency_table(
        df,
        config["label"],
        min_count=1,
    )

    keyness_bigrams_all = create_ngram_frequency_table(
        df,
        config["label"],
        n=2,
        min_count=1,
    )

    keyness_trigrams_all = create_ngram_frequency_table(
        df,
        config["label"],
        n=3,
        min_count=1,
    )

    return {
        "word_counts": word_counts,
        "token_frequencies_all": token_frequencies_all,
        "bigrams_all": bigrams_all,
        "trigrams_all": trigrams_all,
        "_keyness_token_frequencies_all": keyness_token_frequencies_all,
        "_keyness_bigrams_all": keyness_bigrams_all,
        "_keyness_trigrams_all": keyness_trigrams_all,
    }


# ---------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------

def write_results_to_excel(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    """
    Write public frequency and N-gram results to a single Excel workbook.

    Internal keyness source tables are not exported.
    """
    output_file = FREQUENCY_OUTPUT_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        word_count_tables = []
        token_frequency_all_tables = []
        bigram_all_tables = []
        trigram_all_tables = []

        for corpus_name, corpus_results in results.items():
            prefix = CORPUS_CONFIG[corpus_name]["prefix"]

            corpus_results["word_counts"].to_excel(
                writer,
                sheet_name=f"{prefix}_word_counts",
                index=False,
            )

            corpus_results["token_frequencies_all"].to_excel(
                writer,
                sheet_name=f"{prefix}_token_freq_all",
                index=False,
            )

            corpus_results["bigrams_all"].to_excel(
                writer,
                sheet_name=f"{prefix}_bigrams_all",
                index=False,
            )

            corpus_results["trigrams_all"].to_excel(
                writer,
                sheet_name=f"{prefix}_trigrams_all",
                index=False,
            )

            word_count_tables.append(corpus_results["word_counts"])
            token_frequency_all_tables.append(
                corpus_results["token_frequencies_all"]
            )
            bigram_all_tables.append(corpus_results["bigrams_all"])
            trigram_all_tables.append(corpus_results["trigrams_all"])

        if word_count_tables:
            pd.concat(word_count_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_word_counts",
                index=False,
            )

        if token_frequency_all_tables:
            pd.concat(token_frequency_all_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_token_freq_all",
                index=False,
            )

        if bigram_all_tables:
            pd.concat(bigram_all_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_bigrams_all",
                index=False,
            )

        if trigram_all_tables:
            pd.concat(trigram_all_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_trigrams_all",
                index=False,
            )

    print(f"\nSaved frequency and N-gram results to: {output_file}")


def write_keyness_to_excel(keyness_results: dict[str, pd.DataFrame]) -> None:
    """
    Write Direct-vs-Group keyness results to a separate Excel workbook.
    """
    if not keyness_results:
        print("No keyness results to write.")
        return

    KEYNESS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(KEYNESS_OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, df in keyness_results.items():
            if df.empty:
                pd.DataFrame(
                    [{"message": "No results available for this table."}]
                ).to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False,
                )
            else:
                df.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False,
                )

    print(f"Saved keyness results to: {KEYNESS_OUTPUT_FILE}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate word count, token frequency, N-gram, and Direct-vs-Group "
            "keyness tables for the cleaned lexical corpora."
        )
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
        default=3,
        help=(
            "Minimum frequency threshold for public token and N-gram tables, "
            "and minimum total count for keyness tables. Default: 3."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.min_count < 1:
        raise ValueError("--min-count must be at least 1.")

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

    if args.corpus == "both":
        keyness_results = create_keyness_results(
            results,
            min_total_count=args.min_count,
        )
        write_keyness_to_excel(keyness_results)
    else:
        print(
            "Skipping keyness output because --corpus both is required "
            "for Direct-vs-Group comparison."
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())