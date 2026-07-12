#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12b_export_validated_medical_terms_summary.py

Purpose
-------
Create public aggregated summaries of manually validated clinical terminology
mentions from the confidential model-supported terminology review output.

This script is intended as a follow-up to:

    scripts/12_medical_terminology_analysis.py

It compares the communication-level prominence of validated clinical entity
mentions between Direct and Group messages.

The script calculates:
- validated clinical entity mentions per 1,000 cleaned tokens
- percentage of messages containing at least one validated clinical entity mention
- unique validated clinical terms
- Direct-vs-Group comparison overall and by entity type
- a frequency-thresholded public term table
- duplicate-aware counting of model mentions across backends

Important interpretation note
-----------------------------
The outputs describe communication-level terminology density. They do not
represent patient-level disease prevalence, treatment prevalence, or clinical
case counts.

Inputs
------
Confidential review workbook from script 12:

    outputs/confidential/review_files/medical_terminology_model/
    model_entity_candidates_for_review_SJ.xlsx

Expected sheets:
    candidates_for_review
    all_model_mentions

Corpus denominator files:

    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Output
------
Public aggregated workbook:

    outputs/public/tables/medical_terminology/
    validated_clinical_terminology_summary.xlsx

Confidentiality
---------------
The output contains aggregated counts and validated term labels only. It does
not export original messages, KWIC contexts, conversation IDs, message IDs, or
model context snippets.

Duplicate handling
------------------
Model outputs from different backends can detect the same clinical entity in the
same message. For final public counting, the default deduplication mode is
`message_term`: each final validated term/type is counted once per message,
independent of how many models detected it.

Usage
-----
Run from the project root directory after completing manual review:

    python scripts/12b_export_validated_medical_terms_summary.py

Optional:

    python scripts/12b_export_validated_medical_terms_summary.py \
      --min-term-occurrences 3 \
      --min-term-message-count 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_WORKBOOK = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "review_files"
    / "medical_terminology_model"
    / "model_entity_candidates_for_review_SJ.xlsx"
)

DEFAULT_DIRECT_CORPUS = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "cleaned_corpus_tables"
    / "D_utterances_clean_lexical.csv"
)

DEFAULT_GROUP_CORPUS = (
    PROJECT_DIR
    / "outputs"
    / "confidential"
    / "cleaned_corpus_tables"
    / "G_utterances_clean_lexical.csv"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "public"
    / "tables"
    / "medical_terminology"
    / "validated_clinical_terminology_summary.xlsx"
)

DEFAULT_INCLUDE_DECISIONS = {
    "include_diagnosis",
    "include_medication",
    "include_symptom_or_risk",
    "include_procedure",
    "include_therapy",
    "include_treatment",
    "include_clinical_risk",
    "include",
    "included",
    "valid",
    "keep",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export public aggregated Direct-vs-Group summaries of manually "
            "validated clinical terminology mentions."
        )
    )

    parser.add_argument(
        "--review-workbook",
        type=Path,
        default=DEFAULT_REVIEW_WORKBOOK,
        help="Confidential review workbook from script 12.",
    )

    parser.add_argument(
        "--candidates-sheet",
        default="candidates_for_review",
        help="Sheet with manually reviewed candidate entities.",
    )

    parser.add_argument(
        "--mentions-sheet",
        default="all_model_mentions",
        help="Sheet with all model-level entity mentions.",
    )

    parser.add_argument(
        "--direct-corpus",
        type=Path,
        default=DEFAULT_DIRECT_CORPUS,
        help="Direct-message cleaned corpus CSV for denominator counts.",
    )

    parser.add_argument(
        "--group-corpus",
        type=Path,
        default=DEFAULT_GROUP_CORPUS,
        help="Group-message cleaned corpus CSV for denominator counts.",
    )

    parser.add_argument(
        "--text-column",
        default="text_clean_lexical",
        help="Cleaned text column used for token denominators.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Public aggregated output workbook.",
    )

    parser.add_argument(
        "--min-term-occurrences",
        type=int,
        default=3,
        help=(
            "Minimum total validated occurrences for a term to appear in the "
            "public term-level table. Overall/type summaries still use all "
            "included validated mentions. Default: 3."
        ),
    )

    parser.add_argument(
        "--min-term-message-count",
        type=int,
        default=3,
        help=(
            "Minimum number of messages for a term to appear in the public "
            "term-level table. Overall/type summaries still use all included "
            "validated mentions. Default: 3."
        ),
    )

    parser.add_argument(
        "--dedupe-mode",
        choices=["message_term", "span", "none"],
        default="message_term",
        help=(
            "How to handle duplicate model detections after manual validation. "
            "'message_term' counts each final term/type once per message "
            "(recommended); 'span' deduplicates identical message/term/type/"
            "character-span rows; 'none' keeps all validated model rows. "
            "Default: message_term."
        ),
    )

    parser.add_argument(
        "--include-unreviewed-candidates",
        action="store_true",
        help=(
            "Diagnostic option only: include candidates without review_decision. "
            "Do not use for final public outputs."
        ),
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV versions of public summary tables.",
    )

    return parser.parse_args(argv)


def safe_str(value: object) -> str:
    """
    Convert missing values to empty strings and all other values to stripped strings.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_corpus_type(value: object) -> str:
    """
    Normalize corpus labels to direct/group.
    """
    value_norm = safe_str(value).lower()

    if value_norm in {"direct", "dm", "d"}:
        return "direct"

    if value_norm in {"group", "gm", "g"}:
        return "group"

    return value_norm or "unknown"


def simple_tokenize(text: object) -> list[str]:
    """
    Tokenize cleaned text by whitespace.
    """
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    return text.split()


def decision_is_included(
    row: pd.Series,
    include_unreviewed: bool = False,
) -> bool:
    """
    Decide whether a candidate should be included in public aggregate summaries.

    Include by default:
    - review_decision starts with "include"
    - review_decision is one of common keep/valid labels
    - review_decision contains "merge_variant" and final_preferred_term is filled

    Exclude by default:
    - empty review decision
    - exclude_* decisions
    - ambiguous decisions
    """
    decision = safe_str(row.get("review_decision", "")).lower()
    final_preferred_term = safe_str(row.get("final_preferred_term", ""))

    if not decision:
        return include_unreviewed

    if decision.startswith("exclude"):
        return False

    if decision in {"ambiguous", "unclear", "not_sure"}:
        return False

    if decision.startswith("include"):
        return True

    if decision in DEFAULT_INCLUDE_DECISIONS:
        return True

    if "include" in decision:
        return True

    if "merge_variant" in decision and final_preferred_term:
        return True

    return False


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    """
    Raise a clear error if required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {label}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def load_validated_candidates(
    review_workbook: Path,
    candidates_sheet: str,
    include_unreviewed: bool,
) -> pd.DataFrame:
    """
    Load manually reviewed candidate entities and keep included candidates only.
    """
    if not review_workbook.exists():
        raise FileNotFoundError(f"Review workbook not found: {review_workbook}")

    candidates = pd.read_excel(review_workbook, sheet_name=candidates_sheet)

    require_columns(
        candidates,
        ["entity_key", "entity_text", "entity_type_standard"],
        label=f"{review_workbook}:{candidates_sheet}",
    )

    for col in [
        "review_decision",
        "final_entity_type",
        "final_preferred_term",
        "review_note",
    ]:
        if col not in candidates.columns:
            candidates[col] = ""

    candidates = candidates.copy()
    candidates["_include_in_summary"] = candidates.apply(
        lambda row: decision_is_included(
            row,
            include_unreviewed=include_unreviewed,
        ),
        axis=1,
    )

    validated = candidates[candidates["_include_in_summary"]].copy()

    if validated.empty:
        raise ValueError(
            "No validated candidates found. Fill review_decision in the "
            "candidates_for_review sheet, or use --include-unreviewed-candidates "
            "for diagnostic exploration only."
        )

    validated["final_entity_type"] = validated.apply(
        lambda row: safe_str(row.get("final_entity_type", ""))
        or safe_str(row.get("entity_type_standard", "")),
        axis=1,
    )

    validated["final_preferred_term"] = validated.apply(
        lambda row: safe_str(row.get("final_preferred_term", ""))
        or safe_str(row.get("entity_text", "")),
        axis=1,
    )

    # Use exact surface-form matching in addition to entity_key/type. This is
    # important because short clinical abbreviations can share the same normalized
    # key across different meanings, for example "BE" vs "Be" or "PT" vs "pt".
    validated["_entity_text_match"] = validated["entity_text"].apply(safe_str)

    # Fail fast if the same exact model candidate maps to conflicting final terms
    # or entity types. Otherwise the downstream counts would depend on row order.
    conflict_check = (
        validated
        .groupby(
            ["entity_key", "entity_type_standard", "_entity_text_match"],
            dropna=False,
        )
        .agg(
            candidate_rows=("entity_key", "size"),
            final_entity_types=(
                "final_entity_type",
                lambda x: "; ".join(sorted(set(map(safe_str, x)))),
            ),
            final_preferred_terms=(
                "final_preferred_term",
                lambda x: "; ".join(sorted(set(map(safe_str, x)))),
            ),
        )
        .reset_index()
    )

    conflict_rows = []
    for _, conflict_row in conflict_check.iterrows():
        type_values = [
            value
            for value in str(conflict_row["final_entity_types"]).split("; ")
            if value
        ]
        term_values = [
            value
            for value in str(conflict_row["final_preferred_terms"]).split("; ")
            if value
        ]

        if len(set(type_values)) > 1 or len(set(term_values)) > 1:
            conflict_rows.append(conflict_row)

    if conflict_rows:
        conflicts = pd.DataFrame(conflict_rows)
        raise ValueError(
            "Conflicting validated candidate mappings found for identical "
            "entity_key/entity_type_standard/entity_text combinations. "
            "Resolve these in candidates_for_review before running final "
            "summaries. First conflicts:\n"
            f"{conflicts.head(20).to_string(index=False)}"
        )

    duplicated = validated.duplicated(
        subset=["entity_key", "entity_type_standard", "_entity_text_match"],
        keep=False,
    ).sum()

    if duplicated:
        print(
            "WARNING: Duplicate validated candidate mappings detected for "
            f"{duplicated} rows. Keeping first mapping per exact candidate key."
        )

    keep_cols = [
        "entity_key",
        "entity_type_standard",
        "_entity_text_match",
        "entity_text",
        "review_decision",
        "final_entity_type",
        "final_preferred_term",
        "review_note",
    ]

    validated = validated[keep_cols].drop_duplicates(
        subset=["entity_key", "entity_type_standard", "_entity_text_match"],
        keep="first",
    )

    return validated


def load_model_mentions(
    review_workbook: Path,
    mentions_sheet: str,
) -> pd.DataFrame:
    """
    Load all model mention rows from the confidential review workbook.
    """
    mentions = pd.read_excel(review_workbook, sheet_name=mentions_sheet)

    require_columns(
        mentions,
        [
            "entity_key",
            "entity_text",
            "entity_type_standard",
            "corpus_type",
            "message_id",
        ],
        label=f"{review_workbook}:{mentions_sheet}",
    )

    mentions = mentions.copy()
    mentions["corpus_type"] = mentions["corpus_type"].apply(normalize_corpus_type)
    mentions["_entity_text_match"] = mentions["entity_text"].apply(safe_str)
    mentions["message_uid"] = (
        mentions["corpus_type"].astype(str)
        + "__"
        + mentions["message_id"].astype(str)
    )

    return mentions


def load_corpus_size(
    path: Path,
    corpus_type: str,
    text_column: str,
) -> dict[str, object]:
    """
    Load corpus denominator counts for one corpus.
    """
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    text = df[text_column].fillna("").astype(str)
    nonempty = text.str.strip().astype(bool)

    n_messages = int(nonempty.sum())
    n_tokens = int(text[nonempty].apply(lambda x: len(simple_tokenize(x))).sum())

    return {
        "corpus_type": corpus_type,
        "n_messages": n_messages,
        "n_cleaned_tokens": n_tokens,
        "source_file": str(path),
        "text_column": text_column,
    }


def create_corpus_size_table(
    direct_path: Path,
    group_path: Path,
    text_column: str,
) -> pd.DataFrame:
    """
    Create denominator table for Direct and Group corpora.
    """
    return pd.DataFrame(
        [
            load_corpus_size(direct_path, "direct", text_column),
            load_corpus_size(group_path, "group", text_column),
        ]
    )


def attach_manual_validation(
    mentions: pd.DataFrame,
    validated_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join model mentions to manually validated candidate-level decisions.
    """
    validated_mentions = mentions.merge(
        validated_candidates,
        on=["entity_key", "entity_type_standard", "_entity_text_match"],
        how="inner",
        suffixes=("_mention", "_candidate"),
    )

    if validated_mentions.empty:
        raise ValueError(
            "No model mentions matched the validated candidate mappings. "
            "Check entity_key/entity_type_standard columns."
        )

    # Prefer mention-level surface form for variants, but retain final reviewed term/type.
    validated_mentions["final_preferred_term"] = validated_mentions[
        "final_preferred_term"
    ].apply(safe_str)

    validated_mentions["final_entity_type"] = validated_mentions[
        "final_entity_type"
    ].apply(safe_str)

    return validated_mentions


def deduplicate_validated_mentions(
    validated_mentions: pd.DataFrame,
    dedupe_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate validated model mentions before public counting.

    Rationale
    ---------
    Several backends may detect the same clinical entity in the same message.
    The default mode "message_term" counts each final validated term/type once
    per message, independent of backend duplicates. This conservative definition
    is appropriate for communication-level terminology density.
    """
    raw_count = int(len(validated_mentions))

    if dedupe_mode == "none":
        deduped = validated_mentions.copy()
        subset = "none"
    elif dedupe_mode == "span":
        subset_cols = [
            "corpus_type",
            "message_id",
            "final_entity_type",
            "final_preferred_term",
            "start",
            "end",
        ]
        subset = ";".join(subset_cols)
        deduped = validated_mentions.drop_duplicates(subset=subset_cols).copy()
    elif dedupe_mode == "message_term":
        subset_cols = [
            "corpus_type",
            "message_id",
            "final_entity_type",
            "final_preferred_term",
        ]
        subset = ";".join(subset_cols)
        deduped = validated_mentions.drop_duplicates(subset=subset_cols).copy()
    else:
        raise ValueError(f"Unknown dedupe mode: {dedupe_mode}")

    audit = pd.DataFrame(
        [
            {
                "dedupe_mode": dedupe_mode,
                "dedupe_subset": subset,
                "raw_validated_model_rows": raw_count,
                "deduplicated_validated_mentions": int(len(deduped)),
                "duplicate_rows_removed": raw_count - int(len(deduped)),
                "interpretation": (
                    "Counts each final validated term/type once per message."
                    if dedupe_mode == "message_term"
                    else "Counts exact message/term/type/span duplicates once."
                    if dedupe_mode == "span"
                    else "No deduplication applied."
                ),
            }
        ]
    )

    return deduped, audit


def rate_per_1000(n_mentions: int, n_tokens: int) -> float:
    """
    Calculate mentions per 1,000 cleaned tokens.
    """
    if n_tokens <= 0:
        return 0.0

    return n_mentions / n_tokens * 1000


def percent_messages(n_messages_with_entity: int, n_messages_total: int) -> float:
    """
    Calculate percentage of messages with at least one entity mention.
    """
    if n_messages_total <= 0:
        return 0.0

    return n_messages_with_entity / n_messages_total * 100


def create_overall_summary(
    validated_mentions: pd.DataFrame,
    corpus_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Direct-vs-Group overall terminology density summary.
    """
    rows = []

    for _, size_row in corpus_sizes.iterrows():
        corpus = size_row["corpus_type"]
        n_messages = int(size_row["n_messages"])
        n_tokens = int(size_row["n_cleaned_tokens"])

        sub = validated_mentions[
            validated_mentions["corpus_type"].astype(str).eq(corpus)
        ].copy()

        n_mentions = int(len(sub))
        n_messages_with_entity = int(sub["message_uid"].nunique())
        n_unique_terms = int(sub["final_preferred_term"].nunique())

        rows.append(
            {
                "corpus_type": corpus,
                "n_messages": n_messages,
                "n_cleaned_tokens": n_tokens,
                "validated_entity_mentions": n_mentions,
                "mentions_per_1000_cleaned_tokens": rate_per_1000(
                    n_mentions,
                    n_tokens,
                ),
                "messages_with_validated_entity": n_messages_with_entity,
                "percent_messages_with_validated_entity": percent_messages(
                    n_messages_with_entity,
                    n_messages,
                ),
                "unique_validated_terms": n_unique_terms,
            }
        )

    result = pd.DataFrame(rows)

    direct_rate = result.loc[
        result["corpus_type"] == "direct",
        "mentions_per_1000_cleaned_tokens",
    ]

    group_rate = result.loc[
        result["corpus_type"] == "group",
        "mentions_per_1000_cleaned_tokens",
    ]

    if not direct_rate.empty and not group_rate.empty and float(group_rate.iloc[0]) > 0:
        result["direct_vs_group_rate_ratio"] = float(direct_rate.iloc[0]) / float(
            group_rate.iloc[0]
        )
    else:
        result["direct_vs_group_rate_ratio"] = pd.NA

    return result


def create_by_entity_type_long(
    validated_mentions: pd.DataFrame,
    corpus_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create long-format Direct/Group summary by validated entity type.
    """
    entity_types = sorted(validated_mentions["final_entity_type"].dropna().unique())
    rows = []

    for entity_type in entity_types:
        type_df = validated_mentions[
            validated_mentions["final_entity_type"].astype(str).eq(str(entity_type))
        ]

        for _, size_row in corpus_sizes.iterrows():
            corpus = size_row["corpus_type"]
            n_messages = int(size_row["n_messages"])
            n_tokens = int(size_row["n_cleaned_tokens"])

            sub = type_df[type_df["corpus_type"].astype(str).eq(corpus)]

            n_mentions = int(len(sub))
            n_messages_with_entity = int(sub["message_uid"].nunique())
            n_unique_terms = int(sub["final_preferred_term"].nunique())

            rows.append(
                {
                    "final_entity_type": entity_type,
                    "corpus_type": corpus,
                    "n_messages": n_messages,
                    "n_cleaned_tokens": n_tokens,
                    "validated_entity_mentions": n_mentions,
                    "mentions_per_1000_cleaned_tokens": rate_per_1000(
                        n_mentions,
                        n_tokens,
                    ),
                    "messages_with_validated_entity": n_messages_with_entity,
                    "percent_messages_with_validated_entity": percent_messages(
                        n_messages_with_entity,
                        n_messages,
                    ),
                    "unique_validated_terms": n_unique_terms,
                }
            )

    return pd.DataFrame(rows)


def create_by_entity_type_comparison(by_type_long: pd.DataFrame) -> pd.DataFrame:
    """
    Create wide-format Direct-vs-Group comparison by entity type.
    """
    rows = []

    for entity_type, sub in by_type_long.groupby("final_entity_type", dropna=False):
        direct = sub[sub["corpus_type"] == "direct"]
        group = sub[sub["corpus_type"] == "group"]

        def get_value(df: pd.DataFrame, column: str) -> float:
            if df.empty:
                return 0.0
            return float(df[column].iloc[0])

        direct_rate = get_value(direct, "mentions_per_1000_cleaned_tokens")
        group_rate = get_value(group, "mentions_per_1000_cleaned_tokens")

        rate_ratio = direct_rate / group_rate if group_rate > 0 else pd.NA

        rows.append(
            {
                "final_entity_type": entity_type,
                "direct_mentions": int(get_value(direct, "validated_entity_mentions")),
                "group_mentions": int(get_value(group, "validated_entity_mentions")),
                "direct_mentions_per_1000_tokens": direct_rate,
                "group_mentions_per_1000_tokens": group_rate,
                "direct_minus_group_mentions_per_1000_tokens": direct_rate - group_rate,
                "direct_vs_group_rate_ratio": rate_ratio,
                "direct_messages_with_entity": int(
                    get_value(direct, "messages_with_validated_entity")
                ),
                "group_messages_with_entity": int(
                    get_value(group, "messages_with_validated_entity")
                ),
                "direct_percent_messages_with_entity": get_value(
                    direct,
                    "percent_messages_with_validated_entity",
                ),
                "group_percent_messages_with_entity": get_value(
                    group,
                    "percent_messages_with_validated_entity",
                ),
                "direct_unique_terms": int(get_value(direct, "unique_validated_terms")),
                "group_unique_terms": int(get_value(group, "unique_validated_terms")),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            [
                "direct_minus_group_mentions_per_1000_tokens",
                "direct_mentions",
                "group_mentions",
            ],
            ascending=[False, False, False],
        )

    return result


def create_public_term_table(
    validated_mentions: pd.DataFrame,
    min_occurrences: int,
    min_message_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a frequency-thresholded public table of validated terms.

    Rare terms below the public threshold are summarized only in the suppression
    table and are not listed term-by-term.
    """
    group_cols = ["final_entity_type", "final_preferred_term"]

    term_rows = []

    for (entity_type, term), sub in validated_mentions.groupby(group_cols, dropna=False):
        direct = sub[sub["corpus_type"] == "direct"]
        group = sub[sub["corpus_type"] == "group"]

        direct_messages = int(direct["message_uid"].nunique())
        group_messages = int(group["message_uid"].nunique())

        all_message_count = int(sub["message_uid"].nunique())
        total_occurrences = int(len(sub))

        entity_text_variants = "; ".join(
            sorted(set(sub["entity_text_mention"].dropna().astype(str)))
        )

        backends = "; ".join(sorted(set(sub["backend"].dropna().astype(str)))) \
            if "backend" in sub.columns else ""

        term_rows.append(
            {
                "final_entity_type": entity_type,
                "final_preferred_term": term,
                "total_occurrences": total_occurrences,
                "total_message_count": all_message_count,
                "direct_occurrences": int(len(direct)),
                "group_occurrences": int(len(group)),
                "direct_message_count": direct_messages,
                "group_message_count": group_messages,
                "corpus_distribution": (
                    "direct_only"
                    if len(direct) > 0 and len(group) == 0
                    else "group_only"
                    if len(group) > 0 and len(direct) == 0
                    else "direct_and_group"
                ),
                "surface_variants": entity_text_variants,
                "backends": backends,
            }
        )

    all_terms = pd.DataFrame(term_rows)

    if all_terms.empty:
        public_terms = all_terms.copy()
        suppression = pd.DataFrame()
        return public_terms, suppression

    public_mask = (
        (all_terms["total_occurrences"] >= min_occurrences)
        & (all_terms["total_message_count"] >= min_message_count)
    )

    public_terms = all_terms[public_mask].copy()
    suppressed_terms = all_terms[~public_mask].copy()

    public_terms = public_terms.sort_values(
        ["final_entity_type", "total_occurrences", "total_message_count"],
        ascending=[True, False, False],
    )

    if suppressed_terms.empty:
        suppression = pd.DataFrame(
            columns=[
                "final_entity_type",
                "suppressed_terms",
                "suppressed_occurrences",
                "suppressed_message_count_sum",
            ]
        )
    else:
        suppression = (
            suppressed_terms
            .groupby("final_entity_type", dropna=False)
            .agg(
                suppressed_terms=("final_preferred_term", "nunique"),
                suppressed_occurrences=("total_occurrences", "sum"),
                suppressed_message_count_sum=("total_message_count", "sum"),
            )
            .reset_index()
            .sort_values("suppressed_terms", ascending=False)
        )

    return public_terms, suppression


def create_readme_table(
    min_occurrences: int,
    min_message_count: int,
    dedupe_mode: str,
) -> pd.DataFrame:
    """
    Create an explanatory README sheet for the output workbook.
    """
    return pd.DataFrame(
        [
            {
                "field": "purpose",
                "value": (
                    "Public aggregated summary of manually validated clinical "
                    "entity mentions in Direct vs Group messages."
                ),
            },
            {
                "field": "unit_of_analysis",
                "value": (
                    "Communication-level mentions and messages, not patients "
                    "or clinical cases."
                ),
            },
            {
                "field": "main_density_metric",
                "value": "Validated clinical entity mentions per 1,000 cleaned tokens.",
            },
            {
                "field": "message_coverage_metric",
                "value": (
                    "Percentage of messages containing at least one validated "
                    "clinical entity mention."
                ),
            },
            {
                "field": "term_table_threshold",
                "value": (
                    f"Terms are listed publicly only if total_occurrences >= "
                    f"{min_occurrences} and total_message_count >= "
                    f"{min_message_count}."
                ),
            },
            {
                "field": "duplicate_handling",
                "value": (
                    f"Deduplication mode: {dedupe_mode}. The recommended final "
                    "mode is message_term, which counts each final validated "
                    "term/type once per message to avoid duplicate counting "
                    "across model backends."
                ),
            },
            {
                "field": "interpretation_warning",
                "value": (
                    "These metrics do not indicate patient-level prevalence of "
                    "diseases, symptoms, procedures, or treatments."
                ),
            },
        ]
    )


def write_outputs(
    output_file: Path,
    readme: pd.DataFrame,
    corpus_sizes: pd.DataFrame,
    counting_audit: pd.DataFrame,
    overall: pd.DataFrame,
    by_type_long: pd.DataFrame,
    by_type_comparison: pd.DataFrame,
    public_terms: pd.DataFrame,
    suppression: pd.DataFrame,
    write_csv: bool,
) -> None:
    """
    Write public aggregated outputs.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        corpus_sizes.to_excel(writer, sheet_name="corpus_denominators", index=False)
        counting_audit.to_excel(writer, sheet_name="counting_audit", index=False)
        overall.to_excel(writer, sheet_name="overall_direct_group", index=False)
        by_type_long.to_excel(writer, sheet_name="by_type_long", index=False)
        by_type_comparison.to_excel(
            writer,
            sheet_name="by_type_comparison",
            index=False,
        )
        public_terms.to_excel(
            writer,
            sheet_name="validated_terms_public",
            index=False,
        )
        suppression.to_excel(
            writer,
            sheet_name="suppressed_low_freq_terms",
            index=False,
        )

    if write_csv:
        csv_dir = output_file.with_suffix("")
        csv_dir.mkdir(parents=True, exist_ok=True)

        readme.to_csv(csv_dir / "README.csv", index=False, encoding="utf-8")
        corpus_sizes.to_csv(
            csv_dir / "corpus_denominators.csv",
            index=False,
            encoding="utf-8",
        )
        counting_audit.to_csv(
            csv_dir / "counting_audit.csv",
            index=False,
            encoding="utf-8",
        )
        overall.to_csv(
            csv_dir / "overall_direct_group.csv",
            index=False,
            encoding="utf-8",
        )
        by_type_long.to_csv(
            csv_dir / "by_type_long.csv",
            index=False,
            encoding="utf-8",
        )
        by_type_comparison.to_csv(
            csv_dir / "by_type_comparison.csv",
            index=False,
            encoding="utf-8",
        )
        public_terms.to_csv(
            csv_dir / "validated_terms_public.csv",
            index=False,
            encoding="utf-8",
        )
        suppression.to_csv(
            csv_dir / "suppressed_low_freq_terms.csv",
            index=False,
            encoding="utf-8",
        )

    print(f"Saved public aggregated terminology summary to: {output_file}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.min_term_occurrences < 1:
        raise ValueError("--min-term-occurrences must be at least 1.")

    if args.min_term_message_count < 1:
        raise ValueError("--min-term-message-count must be at least 1.")

    print("Loading manually reviewed candidates...")
    validated_candidates = load_validated_candidates(
        review_workbook=args.review_workbook,
        candidates_sheet=args.candidates_sheet,
        include_unreviewed=args.include_unreviewed_candidates,
    )
    print(f"Validated candidate mappings: {len(validated_candidates)}")

    print("Loading model mentions...")
    mentions = load_model_mentions(
        review_workbook=args.review_workbook,
        mentions_sheet=args.mentions_sheet,
    )
    print(f"Model mention rows: {len(mentions)}")

    print("Joining validated candidates to mention rows...")
    validated_mentions_raw = attach_manual_validation(
        mentions=mentions,
        validated_candidates=validated_candidates,
    )
    print(f"Raw validated model mention rows: {len(validated_mentions_raw)}")

    validated_mentions, counting_audit = deduplicate_validated_mentions(
        validated_mentions=validated_mentions_raw,
        dedupe_mode=args.dedupe_mode,
    )
    print(
        "Deduplicated validated mention rows "
        f"({args.dedupe_mode}): {len(validated_mentions)}"
    )

    print("Loading corpus denominators...")
    corpus_sizes = create_corpus_size_table(
        direct_path=args.direct_corpus,
        group_path=args.group_corpus,
        text_column=args.text_column,
    )

    readme = create_readme_table(
        min_occurrences=args.min_term_occurrences,
        min_message_count=args.min_term_message_count,
        dedupe_mode=args.dedupe_mode,
    )

    overall = create_overall_summary(
        validated_mentions=validated_mentions,
        corpus_sizes=corpus_sizes,
    )

    by_type_long = create_by_entity_type_long(
        validated_mentions=validated_mentions,
        corpus_sizes=corpus_sizes,
    )

    by_type_comparison = create_by_entity_type_comparison(by_type_long)

    public_terms, suppression = create_public_term_table(
        validated_mentions=validated_mentions,
        min_occurrences=args.min_term_occurrences,
        min_message_count=args.min_term_message_count,
    )

    write_outputs(
        output_file=args.output_file,
        readme=readme,
        corpus_sizes=corpus_sizes,
        counting_audit=counting_audit,
        overall=overall,
        by_type_long=by_type_long,
        by_type_comparison=by_type_comparison,
        public_terms=public_terms,
        suppression=suppression,
        write_csv=args.write_csv,
    )

    print("\nOverall Direct-vs-Group terminology density:")
    print(overall.to_string(index=False))

    print("\nBy entity type comparison:")
    display_cols = [
        "final_entity_type",
        "direct_mentions_per_1000_tokens",
        "group_mentions_per_1000_tokens",
        "direct_minus_group_mentions_per_1000_tokens",
        "direct_vs_group_rate_ratio",
        "direct_mentions",
        "group_mentions",
    ]
    display_cols = [col for col in display_cols if col in by_type_comparison.columns]
    print(by_type_comparison[display_cols].to_string(index=False))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())