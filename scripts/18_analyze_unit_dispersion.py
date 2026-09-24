#!/usr/bin/env python3
"""Unit-level dispersion analysis for the DocTalk corpus.

This script addresses two related robustness questions:

1. Are workflow markers distributed across many group channels, or are the
   message-level totals concentrated in a few purpose-specific channels?
2. Are selected keyness items distributed across communication units, or are
   pooled token frequencies dominated by a few high-volume units?

The script deliberately separates:

- public aggregate outputs, which contain no communication-unit identifiers;
- a confidential review workbook with anonymized unit-level detail.

Recommended first run
---------------------
Run only the workflow-marker analysis and validate it before enabling the
keyness-item analysis:

python scripts/18_analyze_unit_dispersion.py --analysis workflow

After validation:

python scripts/18_analyze_unit_dispersion.py --analysis all

Expected input
--------------
outputs/confidential/cleaned_corpus_tables/
    utterances_for_collocation_clean_lexical_v2.csv

The input must contain one row per final corpus message and at least:

- conversation_id
- direction
- text_clean_lexical_v2

For group messages, conversation_id represents a Mattermost group channel.
For direct messages, conversation_id represents a dyadic conversation.

Public outputs
--------------
outputs/public/tables/unit_dispersion/
    workflow_marker_channel_dispersion.csv
    keyness_item_unit_dispersion.csv           # with --analysis keyness/all
    unit_dispersion_summary.xlsx

Confidential output
-------------------
outputs/confidential/review_files/unit_dispersion/
    unit_dispersion_detailed_review.xlsx

Important definitions
---------------------
- Marker presence is binary at message level: repeated occurrences within a
  message count once for messages_with_item.
- Token occurrences are retained separately for quality assurance.
- top1/top3 shares refer to the share of marker-positive messages for workflow
  markers and to the share of token occurrences for keyness items.
- Exact whitespace-token matching is used. Substring matching is not used.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Sequence

import pandas as pd


LOGGER = logging.getLogger("unit_dispersion")

DEFAULT_INPUT = Path(
    "outputs/confidential/cleaned_corpus_tables/"
    "utterances_for_collocation_clean_lexical_v2.csv"
)
DEFAULT_MARKER_SOURCE = Path(
    "outputs/public/tables/figure_sources/"
    "marker_weekday_matrix_by_modality_source.csv"
)
DEFAULT_PUBLIC_DIR = Path("outputs/public/tables/unit_dispersion")
DEFAULT_CONFIDENTIAL_DIR = Path(
    "outputs/confidential/review_files/unit_dispersion"
)

DEFAULT_WORKFLOW_MARKERS = [
    "Übergabe",
    "WE",
    "kein_Todo",
    "Rückmeldung",
    "anwesend",
]

DEFAULT_KEYNESS_ITEMS = [
    "du",
    "ich",
    "QuestionMark",
    "Hashtag_PatName",
    "kein_Todo",
    "Mention_KolName",
    "Rückmeldung",
    "anwesend",
]

EXPECTED_MESSAGES = {"direct": 4915, "group": 2547}
EXPECTED_UNITS = {"direct": 293, "group": 86}
# v2 analysis-token counts using the Unicode tokenizer employed by
# the final marker/keyness figure scripts. These differ from the v1
# whitespace denominators used for the medical-terminology analysis.
EXPECTED_ANALYSIS_TOKENS = {"direct": 88287, "group": 90187}

TOKEN_PATTERN = re.compile(
    r"[\wÄÖÜäöüß]+(?:_[\wÄÖÜäöüß]+)*",
    flags=re.UNICODE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze workflow-marker dispersion across group channels and "
            "document frequency of selected keyness items across communication units."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--analysis",
        choices=("workflow", "keyness", "all"),
        default="workflow",
        help=(
            "Analysis component to run. Start with 'workflow'; use 'all' after "
            "validating the workflow output. Default: workflow."
        ),
    )
    parser.add_argument("--unit-col", default="conversation_id")
    parser.add_argument("--direction-col", default="direction")
    parser.add_argument("--text-col", default="text_clean_lexical_v2")
    parser.add_argument(
        "--message-id-col",
        default="id",
        help="Optional unique message identifier used for duplicate checks.",
    )
    parser.add_argument(
        "--workflow-markers",
        default=",".join(DEFAULT_WORKFLOW_MARKERS),
        help="Comma-separated exact tokens.",
    )
    parser.add_argument(
        "--keyness-items",
        default=",".join(DEFAULT_KEYNESS_ITEMS),
        help="Comma-separated exact tokens.",
    )
    parser.add_argument("--marker-source", type=Path, default=DEFAULT_MARKER_SOURCE)
    parser.add_argument("--public-out-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument(
        "--confidential-out-dir", type=Path, default=DEFAULT_CONFIDENTIAL_DIR
    )
    parser.add_argument(
        "--skip-corpus-validation",
        action="store_true",
        help="Only for synthetic/demo data. Never use for the final empirical run.",
    )
    parser.add_argument(
        "--skip-marker-source-validation",
        action="store_true",
        help=(
            "Skip comparison with the existing marker-weekday figure source. "
            "Use only if that file is unavailable."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def parse_item_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one analysis item must be supplied.")
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate analysis items supplied: {items}")
    return items


def normalize_direction(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {
        "direct",
        "dm",
        "direct_message",
        "direct messages",
        "direkt",
        "direktnachrichten",
    }:
        return "direct"
    if normalized in {
        "group",
        "gm",
        "group_message",
        "group messages",
        "gruppe",
        "gruppennachrichten",
    }:
        return "group"
    return normalized


def tokenize_exact(value: object) -> list[str]:
    """Return case-folded, token-exact Unicode tokens from the v2 text."""

    value = "" if pd.isna(value) else str(value)
    return TOKEN_PATTERN.findall(value)

def safe_percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def top_n_share(values: pd.Series, n: int) -> float:
    total = float(values.sum())
    if total == 0:
        return 0.0
    return float(values.nlargest(n).sum()) / total


def distribution_stats(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
        }
    q1 = float(numeric.quantile(0.25))
    median = float(numeric.quantile(0.50))
    q3 = float(numeric.quantile(0.75))
    return {"q1": q1, "median": median, "q3": q3, "iqr": q3 - q1}


def load_corpus(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.input.suffix.lower() == ".csv":
        df = pd.read_csv(args.input)
    elif args.input.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(args.input)
    else:
        raise ValueError("Input must be CSV or Excel.")

    required = {args.unit_col, args.direction_col, args.text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    df = df.copy()
    df[args.direction_col] = df[args.direction_col].map(normalize_direction)
    unexpected = sorted(set(df[args.direction_col].dropna()) - {"direct", "group"})
    if unexpected:
        raise ValueError(f"Unexpected direction labels: {unexpected}")

    if df[args.unit_col].isna().any():
        n_missing = int(df[args.unit_col].isna().sum())
        raise ValueError(f"{n_missing} rows have missing {args.unit_col} values.")

    if args.message_id_col in df.columns and df[args.message_id_col].duplicated().any():
        duplicates = int(df[args.message_id_col].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate message IDs in {args.message_id_col}."
        )

    df["_tokens"] = df[args.text_col].map(tokenize_exact)
    df["_token_count"] = df["_tokens"].map(len)
    return df


def validate_final_corpus(
    df: pd.DataFrame, args: argparse.Namespace
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for modality in ("direct", "group"):
        subset = df.loc[df[args.direction_col] == modality]
        observed_messages = int(len(subset))
        observed_units = int(subset[args.unit_col].nunique())
        observed_tokens = int(subset["_token_count"].sum())

        expected = {
            "messages": EXPECTED_MESSAGES[modality],
            "units": EXPECTED_UNITS[modality],
            "analysis_tokens": EXPECTED_ANALYSIS_TOKENS[modality],
        }
        observed = {
            "messages": observed_messages,
            "units": observed_units,
            "analysis_tokens": observed_tokens,
        }
        for measure in expected:
            passed = observed[measure] == expected[measure]
            rows.append(
                {
                    "check": f"{modality}_{measure}",
                    "observed": observed[measure],
                    "expected": expected[measure],
                    "passed": passed,
                }
            )
            if not passed:
                raise ValueError(
                    f"Corpus validation failed for {modality} {measure}: "
                    f"observed {observed[measure]}, expected {expected[measure]}."
                )
    return rows


def item_counts(tokens: Sequence[str], item: str) -> int:
    return int(sum(token == item for token in tokens))


def build_workflow_dispersion(
    df: pd.DataFrame,
    markers: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_df = df.loc[df[args.direction_col] == "group"].copy()
    if group_df.empty:
        raise ValueError("No group messages found.")

    total_channels = int(group_df[args.unit_col].nunique())
    total_messages = int(len(group_df))
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for marker in markers:
        occurrence_col = f"_occ_{marker}"
        presence_col = f"_present_{marker}"
        group_df[occurrence_col] = group_df["_tokens"].map(
            lambda tokens, m=marker: item_counts(tokens, m)
        )
        group_df[presence_col] = (group_df[occurrence_col] > 0).astype(int)

        by_unit = (
            group_df.groupby(args.unit_col, dropna=False)
            .agg(
                messages_total=(args.unit_col, "size"),
                messages_with_marker=(presence_col, "sum"),
                marker_occurrences=(occurrence_col, "sum"),
            )
            .reset_index()
            .rename(columns={args.unit_col: "unit_id"})
        )
        by_unit["marker"] = marker
        by_unit["modality"] = "group"
        by_unit["channel_prevalence_pct"] = 100.0 * (
            by_unit["messages_with_marker"] / by_unit["messages_total"]
        )
        by_unit = by_unit.sort_values(
            ["messages_with_marker", "channel_prevalence_pct", "messages_total"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        by_unit["marker_rank_by_positive_messages"] = range(1, len(by_unit) + 1)

        message_total = int(by_unit["messages_with_marker"].sum())
        occurrence_total = int(by_unit["marker_occurrences"].sum())
        channels_with_marker = int((by_unit["messages_with_marker"] > 0).sum())
        all_stats = distribution_stats(by_unit["channel_prevalence_pct"])
        positive_stats = distribution_stats(
            by_unit.loc[
                by_unit["messages_with_marker"] > 0, "channel_prevalence_pct"
            ]
        )

        # Internal reconciliation: the per-channel sum must equal the direct
        # message-level binary count.
        message_level_total = int(group_df[presence_col].sum())
        if message_total != message_level_total:
            raise AssertionError(
                f"Marker reconciliation failed for {marker}: "
                f"per-unit sum {message_total} != message-level total "
                f"{message_level_total}."
            )

        top1 = top_n_share(by_unit["messages_with_marker"], 1)
        top3 = top_n_share(by_unit["messages_with_marker"], 3)
        if not (0.0 <= top1 <= top3 <= 1.0):
            raise AssertionError(
                f"Invalid concentration shares for {marker}: top1={top1}, top3={top3}"
            )

        summary_rows.append(
            {
                "marker": marker,
                "modality": "group",
                "group_channels_total": total_channels,
                "group_messages_total": total_messages,
                "messages_with_marker": message_total,
                "message_coverage_pct": safe_percent(message_total, total_messages),
                "marker_occurrences": occurrence_total,
                "channels_with_marker": channels_with_marker,
                "channel_coverage_pct": safe_percent(
                    channels_with_marker, total_channels
                ),
                "top1_channel_share_of_positive_messages_pct": 100.0 * top1,
                "top3_channel_share_of_positive_messages_pct": 100.0 * top3,
                "q1_channel_prevalence_pct_all_channels": all_stats["q1"],
                "median_channel_prevalence_pct_all_channels": all_stats["median"],
                "q3_channel_prevalence_pct_all_channels": all_stats["q3"],
                "iqr_channel_prevalence_pct_all_channels": all_stats["iqr"],
                "median_channel_prevalence_pct_positive_channels": positive_stats[
                    "median"
                ],
            }
        )
        detail_frames.append(by_unit)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True)
    return summary, details


def build_keyness_unit_dispersion(
    df: pd.DataFrame,
    items: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for modality in ("direct", "group"):
        subset = df.loc[df[args.direction_col] == modality].copy()
        total_units = int(subset[args.unit_col].nunique())
        total_messages = int(len(subset))
        total_tokens = int(subset["_token_count"].sum())

        for item in items:
            occurrence_col = f"_occ_{item}"
            presence_col = f"_present_{item}"
            subset[occurrence_col] = subset["_tokens"].map(
                lambda tokens, x=item: item_counts(tokens, x)
            )
            subset[presence_col] = (subset[occurrence_col] > 0).astype(int)

            by_unit = (
                subset.groupby(args.unit_col, dropna=False)
                .agg(
                    messages_total=(args.unit_col, "size"),
                    analysis_tokens=("_token_count", "sum"),
                    messages_with_item=(presence_col, "sum"),
                    item_occurrences=(occurrence_col, "sum"),
                )
                .reset_index()
                .rename(columns={args.unit_col: "unit_id"})
            )
            by_unit["item"] = item
            by_unit["modality"] = modality
            by_unit["message_prevalence_pct"] = 100.0 * (
                by_unit["messages_with_item"] / by_unit["messages_total"]
            )
            by_unit["occurrences_per_1000_tokens"] = by_unit.apply(
                lambda row: (
                    1000.0 * row["item_occurrences"] / row["analysis_tokens"]
                    if row["analysis_tokens"] > 0
                    else 0.0
                ),
                axis=1,
            )
            by_unit = by_unit.sort_values(
                ["item_occurrences", "occurrences_per_1000_tokens", "messages_total"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            by_unit["item_rank_by_occurrences"] = range(1, len(by_unit) + 1)

            units_with_item = int((by_unit["item_occurrences"] > 0).sum())
            messages_with_item = int(by_unit["messages_with_item"].sum())
            occurrences = int(by_unit["item_occurrences"].sum())
            unit_rate_stats = distribution_stats(
                by_unit["occurrences_per_1000_tokens"]
            )
            positive_rate_stats = distribution_stats(
                by_unit.loc[
                    by_unit["item_occurrences"] > 0,
                    "occurrences_per_1000_tokens",
                ]
            )
            top1 = top_n_share(by_unit["item_occurrences"], 1)
            top3 = top_n_share(by_unit["item_occurrences"], 3)

            if int(subset[occurrence_col].sum()) != occurrences:
                raise AssertionError(
                    f"Occurrence reconciliation failed for {item}/{modality}."
                )
            if int(subset[presence_col].sum()) != messages_with_item:
                raise AssertionError(
                    f"Message reconciliation failed for {item}/{modality}."
                )
            if not (0.0 <= top1 <= top3 <= 1.0):
                raise AssertionError(
                    f"Invalid concentration shares for {item}/{modality}."
                )

            summary_rows.append(
                {
                    "item": item,
                    "modality": modality,
                    "units_total": total_units,
                    "units_with_item": units_with_item,
                    "unit_coverage_pct": safe_percent(units_with_item, total_units),
                    "messages_total": total_messages,
                    "messages_with_item": messages_with_item,
                    "message_coverage_pct": safe_percent(
                        messages_with_item, total_messages
                    ),
                    "analysis_tokens": total_tokens,
                    "item_occurrences": occurrences,
                    "occurrences_per_1000_tokens": (
                        1000.0 * occurrences / total_tokens
                        if total_tokens > 0
                        else 0.0
                    ),
                    "top1_unit_share_of_occurrences_pct": 100.0 * top1,
                    "top3_unit_share_of_occurrences_pct": 100.0 * top3,
                    "q1_unit_rate_per_1000_all_units": unit_rate_stats["q1"],
                    "median_unit_rate_per_1000_all_units": unit_rate_stats["median"],
                    "q3_unit_rate_per_1000_all_units": unit_rate_stats["q3"],
                    "iqr_unit_rate_per_1000_all_units": unit_rate_stats["iqr"],
                    "median_unit_rate_per_1000_positive_units": positive_rate_stats[
                        "median"
                    ],
                }
            )
            detail_frames.append(by_unit)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True)
    return summary, details


def validate_against_marker_source(
    workflow_summary: pd.DataFrame,
    marker_source_path: Path,
) -> list[dict[str, object]]:
    if not marker_source_path.exists():
        raise FileNotFoundError(
            f"Marker figure-source file not found: {marker_source_path}. "
            "Use --skip-marker-source-validation only if this file is genuinely unavailable."
        )

    source = pd.read_csv(marker_source_path)
    required = {"direction", "marker", "messages_with_marker", "message_denominator"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            f"Marker source is missing required columns: {sorted(missing)}"
        )
    source = source.copy()
    source["direction"] = source["direction"].map(normalize_direction)
    source = source.loc[source["direction"] == "group"]

    aggregated = (
        source.groupby("marker", as_index=False)
        .agg(
            source_messages_with_marker=("messages_with_marker", "sum"),
            source_message_denominator=("message_denominator", "sum"),
        )
    )

    checks: list[dict[str, object]] = []
    for row in workflow_summary.itertuples(index=False):
        match = aggregated.loc[aggregated["marker"] == row.marker]
        if len(match) != 1:
            raise ValueError(
                f"Expected exactly one aggregated marker-source row for {row.marker}; "
                f"found {len(match)}."
            )
        source_messages = int(match.iloc[0]["source_messages_with_marker"])
        source_denominator = int(match.iloc[0]["source_message_denominator"])
        messages_passed = source_messages == int(row.messages_with_marker)
        denominator_passed = source_denominator == int(row.group_messages_total)
        checks.extend(
            [
                {
                    "check": f"marker_source_messages_{row.marker}",
                    "observed": int(row.messages_with_marker),
                    "expected": source_messages,
                    "passed": messages_passed,
                },
                {
                    "check": f"marker_source_denominator_{row.marker}",
                    "observed": int(row.group_messages_total),
                    "expected": source_denominator,
                    "passed": denominator_passed,
                },
            ]
        )
        if not messages_passed or not denominator_passed:
            raise ValueError(
                f"Marker-source validation failed for {row.marker}: "
                f"messages {row.messages_with_marker}/{source_messages}; "
                f"denominator {row.group_messages_total}/{source_denominator}."
            )
    return checks


def round_public(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    float_columns = result.select_dtypes(include="float").columns
    result[float_columns] = result[float_columns].round(3)
    return result


def save_outputs(
    args: argparse.Namespace,
    workflow_summary: pd.DataFrame | None,
    workflow_details: pd.DataFrame | None,
    keyness_summary: pd.DataFrame | None,
    keyness_details: pd.DataFrame | None,
    validation_rows: Sequence[dict[str, object]],
    markers: Sequence[str],
    keyness_items: Sequence[str],
) -> None:
    args.public_out_dir.mkdir(parents=True, exist_ok=True)
    args.confidential_out_dir.mkdir(parents=True, exist_ok=True)

    public_workflow = (
        round_public(workflow_summary) if workflow_summary is not None else None
    )
    public_keyness = (
        round_public(keyness_summary) if keyness_summary is not None else None
    )
    validation = pd.DataFrame(validation_rows)

    if public_workflow is not None:
        public_workflow.to_csv(
            args.public_out_dir / "workflow_marker_channel_dispersion.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if public_keyness is not None:
        public_keyness.to_csv(
            args.public_out_dir / "keyness_item_unit_dispersion.csv",
            index=False,
            encoding="utf-8-sig",
        )

    public_xlsx = args.public_out_dir / "unit_dispersion_summary.xlsx"
    with pd.ExcelWriter(public_xlsx, engine="openpyxl") as writer:
        if public_workflow is not None:
            public_workflow.to_excel(
                writer, sheet_name="workflow_dispersion", index=False
            )
        if public_keyness is not None:
            public_keyness.to_excel(
                writer, sheet_name="keyness_document_freq", index=False
            )
        validation.to_excel(writer, sheet_name="validation", index=False)

    confidential_xlsx = (
        args.confidential_out_dir / "unit_dispersion_detailed_review.xlsx"
    )
    with pd.ExcelWriter(confidential_xlsx, engine="openpyxl") as writer:
        if workflow_details is not None:
            workflow_details.to_excel(
                writer, sheet_name="workflow_by_unit", index=False
            )
        if keyness_details is not None:
            keyness_details.to_excel(
                writer, sheet_name="keyness_by_unit", index=False
            )
        if public_workflow is not None:
            public_workflow.to_excel(
                writer, sheet_name="workflow_summary", index=False
            )
        if public_keyness is not None:
            public_keyness.to_excel(
                writer, sheet_name="keyness_summary", index=False
            )
        validation.to_excel(writer, sheet_name="validation", index=False)

    config = {
        "input": str(args.input),
        "analysis": args.analysis,
        "unit_col": args.unit_col,
        "direction_col": args.direction_col,
        "text_col": args.text_col,
        "workflow_markers": list(markers),
        "keyness_items": list(keyness_items),
        "marker_source": str(args.marker_source),
        "corpus_validation_skipped": bool(args.skip_corpus_validation),
        "marker_source_validation_skipped": bool(
            args.skip_marker_source_validation
        ),
        "matching": (
            "exact case-sensitive Unicode word-token matching; "
            "underscore compounds preserved"
        ),
        "message_level_presence": "binary; repeated occurrences counted once",
        "public_outputs_contain_unit_identifiers": False,
    }
    with open(
        args.public_out_dir / "unit_dispersion_run_config.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    LOGGER.info("Public outputs written to %s", args.public_out_dir)
    LOGGER.info("Confidential review output written to %s", confidential_xlsx)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(levelname)s: %(message)s",
    )

    markers = parse_item_list(args.workflow_markers)
    keyness_items = parse_item_list(args.keyness_items)
    df = load_corpus(args)
    LOGGER.info("Loaded %s messages from %s", len(df), args.input)

    validation_rows: list[dict[str, object]] = []
    if args.skip_corpus_validation:
        LOGGER.warning(
            "Final corpus validation was skipped. Do not use this option for "
            "the empirical publication run."
        )
    else:
        validation_rows.extend(validate_final_corpus(df, args))

    workflow_summary: pd.DataFrame | None = None
    workflow_details: pd.DataFrame | None = None
    keyness_summary: pd.DataFrame | None = None
    keyness_details: pd.DataFrame | None = None

    if args.analysis in {"workflow", "all"}:
        workflow_summary, workflow_details = build_workflow_dispersion(
            df, markers, args
        )
        if args.skip_marker_source_validation:
            LOGGER.warning(
                "Marker figure-source validation was skipped. Inspect totals manually."
            )
        else:
            validation_rows.extend(
                validate_against_marker_source(
                    workflow_summary, args.marker_source
                )
            )

    if args.analysis in {"keyness", "all"}:
        keyness_summary, keyness_details = build_keyness_unit_dispersion(
            df, keyness_items, args
        )

    save_outputs(
        args=args,
        workflow_summary=workflow_summary,
        workflow_details=workflow_details,
        keyness_summary=keyness_summary,
        keyness_details=keyness_details,
        validation_rows=validation_rows,
        markers=markers,
        keyness_items=keyness_items,
    )

    if workflow_summary is not None:
        print("\nWorkflow-marker channel dispersion")
        print(round_public(workflow_summary).to_string(index=False))
    if keyness_summary is not None:
        print("\nKeyness-item unit dispersion")
        print(round_public(keyness_summary).to_string(index=False))
    print("\nAll requested analyses completed successfully.")


if __name__ == "__main__":
    main()
