#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_frequency_ngram_keyness.py

Purpose
-------
Generate lexical frequency and N-gram tables for the cleaned direct-message
and group-message corpora.

This script calculates:
- word counts before and after lexical cleaning
- token frequencies, all tokens
- token frequencies, content tokens after content-stopword filtering
- token frequencies, interaction/style tokens after interaction-stopword filtering
- bigrams, all tokens
- bigrams, content tokens
- bigrams, interaction/style tokens
- trigrams, all tokens
- trigrams, content tokens
- trigrams, interaction/style tokens

Important methodological note
-----------------------------
For token frequencies, stopwords are removed before counting tokens.

For N-grams in filtered views such as content and interaction, N-grams are
created from the full cleaned token sequence first. Afterwards, N-grams
containing stopwords are excluded. This preserves original token adjacency
and avoids artificial N-grams caused by removing stopwords before N-gram
generation.

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

Rule files
----------
Expected rule files:

    rules/stopwords_content.txt
    rules/stopwords_interaction.txt
    rules/protected_tokens.txt

All files should contain one token per line.
Empty lines and lines starting with '#' are ignored.

Outputs
-------
Aggregated tables are written to:

    outputs/public/tables/frequency_ngram_results.xlsx

Usage
-----
Run from the project root directory:

    python scripts/04_frequency_ngram_keyness.py --corpus direct
    python scripts/04_frequency_ngram_keyness.py --corpus group
    python scripts/04_frequency_ngram_keyness.py --corpus both

Optional:

    python scripts/04_frequency_ngram_keyness.py --corpus both --min-count 3
"""

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


RULES_DIR = PROJECT_DIR / "rules"

STOPWORDS_CONTENT_FILE = RULES_DIR / "stopwords_content.txt"
STOPWORDS_INTERACTION_FILE = RULES_DIR / "stopwords_interaction.txt"
PROTECTED_TOKENS_FILE = RULES_DIR / "protected_tokens.txt"

# Fallback for older project state
LEGACY_STOPWORDS_FILE = RULES_DIR / "stopwords_custom.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
# Default protected analytical tokens
# ---------------------------------------------------------------------

DEFAULT_PROTECTED_TOKENS = {
    "PatName",
    "KolName",
    "Mention_KolName",
    "Mention_All",
    "Hashtag_PatName",
    "Todo",
    "kein_Todo",
    "Datum",
    "Monat",
    "Jahr",
    "Klinik",
    "Klinikstandort",
    "Telefonnummer",
    "Link",
    "Link_intern",
    "Ort",
    "Station",
    "Übergabe",
    "Rückmeldung",
    "Patient",
    "Mitpatient",
    "Probatorik_Patient",
    "Therapeut",
    "Bezugstherapeut",
    "Kreativtherapeut",
    "Psychotherapeut",
    "Behandler",
    "Behandlerwechsel",
    "Arzt",
    "Psychologe",
    "Psychiater",
    "Freund",
}


# ---------------------------------------------------------------------
# Rule loading helpers
# ---------------------------------------------------------------------

def load_token_list(token_file: Path, label: str) -> set[str]:
    """
    Load a token list from a text file.

    One token per line. Empty lines and lines starting with '#'
    are ignored.
    """

    if not token_file.exists():
        print(f"Warning: {label} file not found: {token_file}")
        return set()

    tokens = set()

    with open(token_file, "r", encoding="utf-8") as file:
        for line in file:
            token = line.strip()

            if not token:
                continue

            if token.startswith("#"):
                continue

            tokens.add(token)

    print(f"Loaded {len(tokens)} entries from {label}: {token_file}")

    return tokens


def load_stopwords(stopwords_file: Path, label: str) -> set[str]:
    """
    Load stopwords and normalize them to lowercase for case-insensitive
    stopword filtering.
    """

    stopwords = load_token_list(stopwords_file, label)
    return {word.lower() for word in stopwords}


def load_protected_tokens(protected_tokens_file: Path) -> set[str]:
    """
    Load protected analytical tokens.

    If no protected token file exists, use the default protected token set.
    """

    protected_tokens = load_token_list(
        protected_tokens_file,
        label="protected tokens",
    )

    if not protected_tokens:
        protected_tokens = DEFAULT_PROTECTED_TOKENS
        print(
            "Using default protected tokens because no protected token file "
            "was found or the file was empty."
        )

    return protected_tokens


def load_rule_sets() -> tuple[set[str], set[str], set[str]]:
    """
    Load content stopwords, interaction stopwords, and protected tokens.

    If the new content/interaction stopword files do not exist yet,
    the script falls back to rules/stopwords_custom.txt for both views.
    This keeps the script compatible with the previous project state.
    """

    protected_tokens = load_protected_tokens(PROTECTED_TOKENS_FILE)

    if STOPWORDS_CONTENT_FILE.exists():
        stopwords_content = load_stopwords(
            STOPWORDS_CONTENT_FILE,
            label="content stopwords",
        )
    else:
        print(
            f"Warning: content stopword file not found: "
            f"{STOPWORDS_CONTENT_FILE}"
        )
        print(f"Trying fallback file: {LEGACY_STOPWORDS_FILE}")
        stopwords_content = load_stopwords(
            LEGACY_STOPWORDS_FILE,
            label="legacy/content stopwords",
        )

    if STOPWORDS_INTERACTION_FILE.exists():
        stopwords_interaction = load_stopwords(
            STOPWORDS_INTERACTION_FILE,
            label="interaction stopwords",
        )
    else:
        print(
            f"Warning: interaction stopword file not found: "
            f"{STOPWORDS_INTERACTION_FILE}"
        )
        print(f"Trying fallback file: {LEGACY_STOPWORDS_FILE}")
        stopwords_interaction = load_stopwords(
            LEGACY_STOPWORDS_FILE,
            label="legacy/interaction stopwords",
        )

    return stopwords_content, stopwords_interaction, protected_tokens


# ---------------------------------------------------------------------
# Tokenization and filtering helpers
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


def token_is_allowed_in_view(
    token: str,
    stopwords: set[str],
    protected_tokens: set[str],
) -> bool:
    """
    Check whether a token should be kept in a filtered analytical view.

    Protected analytical tokens are always allowed.
    Stopword matching is case-insensitive.
    """

    if token in protected_tokens:
        return True

    if token.lower() in stopwords:
        return False

    return True


def filter_tokens(
    tokens: list[str],
    stopwords: set[str],
    protected_tokens: set[str],
) -> list[str]:
    """
    Remove stopwords from a token list while preserving standardized
    analytical tokens.

    Stopword matching is case-insensitive.
    Protected-token matching is case-sensitive to preserve standardized
    analytical markers exactly as generated by the cleaning script.
    """

    return [
        token
        for token in tokens
        if token_is_allowed_in_view(
            token,
            stopwords=stopwords,
            protected_tokens=protected_tokens,
        )
    ]


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


# ---------------------------------------------------------------------
# Frequency and N-gram helpers
# ---------------------------------------------------------------------

def create_token_frequency_table(
    df: pd.DataFrame,
    corpus_label: str,
    view: str,
    min_count: int = 1,
    stopwords: set[str] | None = None,
    protected_tokens: set[str] | None = None,
) -> pd.DataFrame:
    """
    Create token frequency table from cleaned text.

    For filtered views, stopwords are removed before token counting.

    view can be:
    - all
    - content
    - interaction
    """

    if stopwords is None:
        stopwords = set()

    if protected_tokens is None:
        protected_tokens = set()

    counter = Counter()

    for text in df["text_clean_lexical"].fillna("").astype(str):
        tokens = simple_tokenize(text)

        if view != "all":
            tokens = filter_tokens(
                tokens,
                stopwords=stopwords,
                protected_tokens=protected_tokens,
            )

        counter.update(tokens)

    records = []
    total_tokens = sum(counter.values())

    for token, count in counter.items():
        if count >= min_count:
            records.append(
                {
                    "corpus": corpus_label,
                    "view": view,
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
    view: str,
    n: int,
    min_count: int = 1,
    stopwords: set[str] | None = None,
    protected_tokens: set[str] | None = None,
) -> pd.DataFrame:
    """
    Create N-gram frequency table from cleaned text.

    For the all-token view, N-grams are built from all cleaned tokens.

    For filtered views such as content and interaction, N-grams are first
    built from the full cleaned token sequence and only then filtered.
    This preserves original adjacency and avoids artificial N-grams that
    would be created by removing stopwords before N-gram generation.

    view can be:
    - all
    - content
    - interaction
    """

    if stopwords is None:
        stopwords = set()

    if protected_tokens is None:
        protected_tokens = set()

    counter = Counter()

    for text in df["text_clean_lexical"].fillna("").astype(str):
        tokens = simple_tokenize(text)
        ngrams = make_ngrams(tokens, n)

        if view != "all":
            ngrams = [
                ngram
                for ngram in ngrams
                if all(
                    token_is_allowed_in_view(
                        token,
                        stopwords=stopwords,
                        protected_tokens=protected_tokens,
                    )
                    for token in ngram
                )
            ]

        counter.update(ngrams)

    records = []
    total_ngrams = sum(counter.values())

    for ngram_tuple, count in counter.items():
        if count >= min_count:
            ngram = " ".join(ngram_tuple)

            records.append(
                {
                    "corpus": corpus_label,
                    "view": view,
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
    view: str,
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

        direct_freq_per_1000 = (
            direct_count / direct_total * 1000
            if direct_total > 0
            else 0
        )

        group_freq_per_1000 = (
            group_count / group_total * 1000
            if group_total > 0
            else 0
        )

        difference_per_1000 = direct_freq_per_1000 - group_freq_per_1000

        # Smoothed relative frequencies
        direct_prop_smoothed = (
            (direct_count + smoothing) / (direct_total + smoothing)
            if direct_total > 0
            else 0
        )

        group_prop_smoothed = (
            (group_count + smoothing) / (group_total + smoothing)
            if group_total > 0
            else 0
        )

        log_ratio = _safe_log2(direct_prop_smoothed / group_prop_smoothed)

        # Smoothed odds ratio
        direct_non_count = max(direct_total - direct_count, 0)
        group_non_count = max(group_total - group_count, 0)

        direct_odds = (direct_count + smoothing) / (direct_non_count + smoothing)
        group_odds = (group_count + smoothing) / (group_non_count + smoothing)

        odds_ratio_smoothed = direct_odds / group_odds if group_odds > 0 else 0

        # Log-likelihood G2
        observed_total = direct_count + group_count
        corpus_total = direct_total + group_total

        expected_direct = (
            direct_total * observed_total / corpus_total
            if corpus_total > 0
            else 0
        )

        expected_group = (
            group_total * observed_total / corpus_total
            if corpus_total > 0
            else 0
        )

        log_likelihood = 0.0

        if direct_count > 0 and expected_direct > 0:
            log_likelihood += direct_count * math.log(direct_count / expected_direct)

        if group_count > 0 and expected_group > 0:
            log_likelihood += group_count * math.log(group_count / expected_group)

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
                "view": view,
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
    Create Direct-vs-Group keyness tables for selected analytical views.

    Keyness is computed for:
    - content tokens
    - interaction tokens
    - content bigrams
    - interaction bigrams
    - content trigrams
    - interaction trigrams
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
            "name": "key_content_tokens",
            "direct_key": "token_frequencies_content",
            "group_key": "token_frequencies_content",
            "item_column": "token",
            "item_type": "token",
            "view": "content",
        },
        {
            "name": "key_interaction_tokens",
            "direct_key": "token_frequencies_interaction",
            "group_key": "token_frequencies_interaction",
            "item_column": "token",
            "item_type": "token",
            "view": "interaction",
        },
        {
            "name": "key_content_bigrams",
            "direct_key": "bigrams_content",
            "group_key": "bigrams_content",
            "item_column": "ngram",
            "item_type": "bigram",
            "view": "content",
        },
        {
            "name": "key_interaction_bigrams",
            "direct_key": "bigrams_interaction",
            "group_key": "bigrams_interaction",
            "item_column": "ngram",
            "item_type": "bigram",
            "view": "interaction",
        },
        {
            "name": "key_content_trigrams",
            "direct_key": "trigrams_content",
            "group_key": "trigrams_content",
            "item_column": "ngram",
            "item_type": "trigram",
            "view": "content",
        },
        {
            "name": "key_interaction_trigrams",
            "direct_key": "trigrams_interaction",
            "group_key": "trigrams_interaction",
            "item_column": "ngram",
            "item_type": "trigram",
            "view": "interaction",
        },
    ]

    keyness_results = {}

    for spec in specs:
        keyness_results[spec["name"]] = compute_keyness_table(
            direct_df=direct_results[spec["direct_key"]],
            group_df=group_results[spec["group_key"]],
            item_column=spec["item_column"],
            item_type=spec["item_type"],
            view=spec["view"],
            min_total_count=min_total_count,
        )

    return keyness_results

# ---------------------------------------------------------------------
# Main corpus processing
# ---------------------------------------------------------------------

def analyze_corpus(
    corpus_name: str,
    min_count: int,
    stopwords_content: set[str],
    stopwords_interaction: set[str],
    protected_tokens: set[str],
) -> dict[str, pd.DataFrame]:
    """
    Run word count, token frequency, bigram, and trigram analyses for one corpus.
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
        view="all",
        min_count=min_count,
        stopwords=None,
        protected_tokens=protected_tokens,
    )

    token_frequencies_content = create_token_frequency_table(
        df,
        config["label"],
        view="content",
        min_count=min_count,
        stopwords=stopwords_content,
        protected_tokens=protected_tokens,
    )

    token_frequencies_interaction = create_token_frequency_table(
        df,
        config["label"],
        view="interaction",
        min_count=min_count,
        stopwords=stopwords_interaction,
        protected_tokens=protected_tokens,
    )

    bigrams_all = create_ngram_frequency_table(
        df,
        config["label"],
        view="all",
        n=2,
        min_count=min_count,
        stopwords=None,
        protected_tokens=protected_tokens,
    )

    bigrams_content = create_ngram_frequency_table(
        df,
        config["label"],
        view="content",
        n=2,
        min_count=min_count,
        stopwords=stopwords_content,
        protected_tokens=protected_tokens,
    )

    bigrams_interaction = create_ngram_frequency_table(
        df,
        config["label"],
        view="interaction",
        n=2,
        min_count=min_count,
        stopwords=stopwords_interaction,
        protected_tokens=protected_tokens,
    )

    trigrams_all = create_ngram_frequency_table(
        df,
        config["label"],
        view="all",
        n=3,
        min_count=min_count,
        stopwords=None,
        protected_tokens=protected_tokens,
    )

    trigrams_content = create_ngram_frequency_table(
        df,
        config["label"],
        view="content",
        n=3,
        min_count=min_count,
        stopwords=stopwords_content,
        protected_tokens=protected_tokens,
    )

    trigrams_interaction = create_ngram_frequency_table(
        df,
        config["label"],
        view="interaction",
        n=3,
        min_count=min_count,
        stopwords=stopwords_interaction,
        protected_tokens=protected_tokens,
    )

    return {
        "word_counts": word_counts,
        "token_frequencies_all": token_frequencies_all,
        "token_frequencies_content": token_frequencies_content,
        "token_frequencies_interaction": token_frequencies_interaction,
        "bigrams_all": bigrams_all,
        "bigrams_content": bigrams_content,
        "bigrams_interaction": bigrams_interaction,
        "trigrams_all": trigrams_all,
        "trigrams_content": trigrams_content,
        "trigrams_interaction": trigrams_interaction,
    }


# ---------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------

def write_results_to_excel(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    """
    Write all results to a single Excel workbook with separate sheets.
    """

    output_file = OUTPUT_DIR / "frequency_ngram_results.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        word_count_tables = []

        token_frequency_all_tables = []
        token_frequency_content_tables = []
        token_frequency_interaction_tables = []

        bigram_all_tables = []
        bigram_content_tables = []
        bigram_interaction_tables = []

        trigram_all_tables = []
        trigram_content_tables = []
        trigram_interaction_tables = []

        for corpus_name, corpus_results in results.items():
            prefix = CORPUS_CONFIG[corpus_name]["prefix"]

            # Individual corpus sheets
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

            corpus_results["token_frequencies_content"].to_excel(
                writer,
                sheet_name=f"{prefix}_token_freq_content",
                index=False,
            )

            corpus_results["token_frequencies_interaction"].to_excel(
                writer,
                sheet_name=f"{prefix}_token_freq_interaction",
                index=False,
            )

            corpus_results["bigrams_all"].to_excel(
                writer,
                sheet_name=f"{prefix}_bigrams_all",
                index=False,
            )

            corpus_results["bigrams_content"].to_excel(
                writer,
                sheet_name=f"{prefix}_bigrams_content",
                index=False,
            )

            corpus_results["bigrams_interaction"].to_excel(
                writer,
                sheet_name=f"{prefix}_bigrams_interaction",
                index=False,
            )

            corpus_results["trigrams_all"].to_excel(
                writer,
                sheet_name=f"{prefix}_trigrams_all",
                index=False,
            )

            corpus_results["trigrams_content"].to_excel(
                writer,
                sheet_name=f"{prefix}_trigrams_content",
                index=False,
            )

            corpus_results["trigrams_interaction"].to_excel(
                writer,
                sheet_name=f"{prefix}_trigrams_interaction",
                index=False,
            )

            # Collect combined tables
            word_count_tables.append(corpus_results["word_counts"])

            token_frequency_all_tables.append(
                corpus_results["token_frequencies_all"]
            )
            token_frequency_content_tables.append(
                corpus_results["token_frequencies_content"]
            )
            token_frequency_interaction_tables.append(
                corpus_results["token_frequencies_interaction"]
            )

            bigram_all_tables.append(corpus_results["bigrams_all"])
            bigram_content_tables.append(corpus_results["bigrams_content"])
            bigram_interaction_tables.append(corpus_results["bigrams_interaction"])

            trigram_all_tables.append(corpus_results["trigrams_all"])
            trigram_content_tables.append(corpus_results["trigrams_content"])
            trigram_interaction_tables.append(
                corpus_results["trigrams_interaction"]
            )

        # Combined sheets
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

        if token_frequency_content_tables:
            pd.concat(token_frequency_content_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_token_freq_content",
                index=False,
            )

        if token_frequency_interaction_tables:
            pd.concat(
                token_frequency_interaction_tables,
                ignore_index=True,
            ).to_excel(
                writer,
                sheet_name="combined_token_freq_interaction",
                index=False,
            )

        if bigram_all_tables:
            pd.concat(bigram_all_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_bigrams_all",
                index=False,
            )

        if bigram_content_tables:
            pd.concat(bigram_content_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_bigrams_content",
                index=False,
            )

        if bigram_interaction_tables:
            pd.concat(bigram_interaction_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_bigrams_interaction",
                index=False,
            )

        if trigram_all_tables:
            pd.concat(trigram_all_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_trigrams_all",
                index=False,
            )

        if trigram_content_tables:
            pd.concat(trigram_content_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_trigrams_content",
                index=False,
            )

        if trigram_interaction_tables:
            pd.concat(trigram_interaction_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="combined_trigrams_interaction",
                index=False,
            )

    print(f"\nSaved results to: {output_file}")

def write_keyness_to_excel(keyness_results: dict[str, pd.DataFrame]) -> None:
    """
    Write Direct-vs-Group keyness results to a separate Excel workbook.
    """

    if not keyness_results:
        print("No keyness results to write.")
        return

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate word count, token frequency, and N-gram tables "
            "for all-token, content, and interaction/style views."
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
            "Minimum frequency threshold for tokens and N-grams. "
            "Default: 3 to avoid exporting rare text-adjacent N-grams "
            "to public output tables."
        ),
    )

    args = parser.parse_args()

    (
        stopwords_content,
        stopwords_interaction,
        protected_tokens,
    ) = load_rule_sets()

    if args.corpus == "both":
        selected_corpora = ["direct", "group"]
    else:
        selected_corpora = [args.corpus]

    results = {}

    for corpus_name in selected_corpora:
        results[corpus_name] = analyze_corpus(
            corpus_name,
            min_count=args.min_count,
            stopwords_content=stopwords_content,
            stopwords_interaction=stopwords_interaction,
            protected_tokens=protected_tokens,
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


if __name__ == "__main__":
    main()