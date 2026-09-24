#!/usr/bin/env python3
"""
Team-sensitivity analysis for the DocTalk corpus.

This script compares the published Group corpus with two restricted variants:

1. group_full
   All Group channels.
2. group_nonadjacent
   Psychosomatic and unattributed channels, excluding channels explicitly
   assigned to adjacent clinical teams.
3. group_psychosomatic
   Only channels explicitly classified as psychosomatic_core.

The Direct corpus is unchanged and serves as the reference corpus for the
keyness comparisons.

The script deliberately separates confidential classification work from
public aggregate outputs. Original team names and channel names are written
only to outputs/confidential/.

Recommended workflow
--------------------

Step 1: create the confidential classification template

    python scripts/19_team_sensitivity_analysis.py --mode inventory

Open the generated workbook and complete the ``category`` column with exactly
one of:

    psychosomatic_core
    adjacent_clinical_team
    unattributed

Step 2: run the analysis

    python scripts/19_team_sensitivity_analysis.py \
        --mode analyze \
        --classification-file \
        outputs/confidential/review_files/team_sensitivity/\
        team_channel_classification.xlsx

Optional medical-terminology sensitivity analysis
--------------------------------------------------

Pass a message-level file containing manually validated entity mentions:

    python scripts/19_team_sensitivity_analysis.py \
        --mode analyze \
        --classification-file <classification.xlsx> \
        --medical-mentions <validated_mentions.xlsx>

The medical file must contain a message identifier. The script detects common
column names automatically, or they can be supplied explicitly with
``--medical-message-id-col``, ``--medical-term-col`` and
``--medical-type-col``.

Important methodological notes
------------------------------

- Group communication units are Mattermost channels, not reply threads.
- Keyness uses the whitespace-tokenized v2 cleaned lexical text and preserves
  case, matching the principal lexical analysis rather than the case-folded
  unit-dispersion robustness analysis.
- Workflow markers are matched as exact cleaned tokens.
- G2 is a descriptive ranking measure, not an inferential population test.
- No user-level analysis is attempted.
- Public outputs contain aggregate values only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


LOGGER = logging.getLogger("team_sensitivity")

DEFAULT_INPUT = Path(
    "outputs/confidential/cleaned_corpus_tables/"
    "utterances_for_collocation_clean_lexical_v2.csv"
)
DEFAULT_CLASSIFICATION = Path(
    "outputs/confidential/review_files/team_sensitivity/"
    "team_channel_classification.xlsx"
)
DEFAULT_PUBLIC_DIR = Path("outputs/public/tables/team_sensitivity")
DEFAULT_CONFIDENTIAL_DIR = Path(
    "outputs/confidential/review_files/team_sensitivity"
)

EXPECTED_MESSAGES = {"direct": 4915, "group": 2547}
EXPECTED_UNITS = {"direct": 293, "group": 86}
EXPECTED_FULL_GROUP_MARKER_MESSAGES = {
    "Übergabe": 199,
    "WE": 91,
    "kein_Todo": 245,
    "Rückmeldung": 214,
    "anwesend": 175,
}

VALID_CATEGORIES = (
    "psychosomatic_core",
    "adjacent_clinical_team",
    "unattributed",
)

DEFAULT_MARKERS = (
    "Übergabe",
    "WE",
    "kein_Todo",
    "Rückmeldung",
    "anwesend",
)

DEFAULT_KEY_ITEMS = (
    "du",
    "ich",
    "QuestionMark",
    "PatName",
    "Hashtag_PatName",
    "kein_Todo",
    "Mention_KolName",
    "Rückmeldung",
    "anwesend",
)

# Exact Unicode tokenization used by the final marker and unit-dispersion
# scripts. This is deliberately separate from the whitespace tokenization
# used for the principal lexical/keyness denominators.
TOKEN_PATTERN = re.compile(
    r"[\wÄÖÜäöüß]+(?:_[\wÄÖÜäöüß]+)*",
    flags=re.UNICODE,
)

WEEKDAY_ORDER = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

WEEKDAY_MAP = {
    "monday": "Monday",
    "mon": "Monday",
    "montag": "Monday",
    "dienstag": "Tuesday",
    "tuesday": "Tuesday",
    "tue": "Tuesday",
    "mittwoch": "Wednesday",
    "wednesday": "Wednesday",
    "wed": "Wednesday",
    "donnerstag": "Thursday",
    "thursday": "Thursday",
    "thu": "Thursday",
    "freitag": "Friday",
    "friday": "Friday",
    "fri": "Friday",
    "samstag": "Saturday",
    "saturday": "Saturday",
    "sat": "Saturday",
    "sonntag": "Sunday",
    "sunday": "Sunday",
    "sun": "Sunday",
}


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    classification_file: str | None
    text_col: str
    marker_text_col: str
    message_id_col: str
    unit_col: str
    direction_col: str
    team_col: str
    channel_col: str
    weekday_col: str
    timestamp_col: str
    min_total_frequency: int
    markers: list[str]
    selected_key_items: list[str]
    medical_mentions: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a confidential team-classification template or run the "
            "DocTalk Group team-sensitivity analysis."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("inventory", "analyze"),
        required=True,
        help="Create the classification workbook or run the analysis.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--classification-file", type=Path, default=DEFAULT_CLASSIFICATION
    )
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument(
        "--confidential-dir", type=Path, default=DEFAULT_CONFIDENTIAL_DIR
    )
    parser.add_argument("--text-col", default="text_clean_lexical")
    parser.add_argument(
        "--marker-text-col",
        default="text_clean_lexical_v2",
        help=(
            "Analysis-specific v2 text used for workflow-marker matching. "
            "This is intentionally separate from --text-col, which retains "
            "the published whitespace-based keyness denominators."
        ),
    )
    parser.add_argument("--message-id-col", default="id")
    parser.add_argument("--unit-col", default="conversation_id")
    parser.add_argument("--direction-col", default="direction")
    parser.add_argument("--team-col", default="Team Name")
    parser.add_argument("--channel-col", default="Channel Name")
    parser.add_argument("--weekday-col", default="weekday")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--min-total-frequency", type=int, default=3)
    parser.add_argument(
        "--markers",
        nargs="+",
        default=list(DEFAULT_MARKERS),
        help="Exact cleaned tokens used as workflow markers.",
    )
    parser.add_argument(
        "--selected-key-items",
        nargs="+",
        default=list(DEFAULT_KEY_ITEMS),
        help="Exact case-sensitive tokens highlighted in the sensitivity table.",
    )
    parser.add_argument(
        "--medical-mentions",
        type=Path,
        default=None,
        help="Optional CSV/XLSX containing manually validated entity mentions.",
    )
    parser.add_argument("--medical-sheet", default=None)
    parser.add_argument("--medical-message-id-col", default=None)
    parser.add_argument("--medical-term-col", default=None)
    parser.add_argument("--medical-type-col", default=None)
    parser.add_argument(
        "--skip-corpus-validation",
        action="store_true",
        help="Diagnostic only. Do not use for the publication run.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def clean_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_direction(series: pd.Series) -> pd.Series:
    return clean_string(series).str.casefold()


def tokenize_whitespace(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return str(value).split()


def tokenize_exact(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return TOKEN_PATTERN.findall(str(value))


def load_corpus(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.input}")
    df = pd.read_csv(args.input)
    require_columns(
        df,
        (
            args.text_col,
            args.marker_text_col,
            args.message_id_col,
            args.unit_col,
            args.direction_col,
        ),
        "Corpus",
    )
    df = df.copy()
    df["_direction"] = normalize_direction(df[args.direction_col])
    unexpected = sorted(set(df["_direction"]) - {"direct", "group"})
    if unexpected:
        raise ValueError(f"Unexpected direction values: {unexpected}")
    df["_message_id"] = clean_string(df[args.message_id_col])
    df["_unit_id"] = clean_string(df[args.unit_col])
    if (df["_message_id"] == "").any():
        raise ValueError("Corpus contains empty message identifiers.")
    if (df["_unit_id"] == "").any():
        raise ValueError("Corpus contains empty communication-unit identifiers.")
    if df["_message_id"].duplicated().any():
        duplicated = int(df["_message_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Message identifiers are not unique; {duplicated} rows are duplicated."
        )
    # Principal lexical/keyness representation and denominator.
    df["_tokens"] = df[args.text_col].map(tokenize_whitespace)
    df["_token_count"] = df["_tokens"].map(len)
    # Exact Unicode tokens used only for workflow-marker detection. Keeping
    # this separate prevents accidental replacement of the published
    # whitespace-based lexical denominators.
    df["_marker_tokens"] = df[args.marker_text_col].map(tokenize_exact)
    LOGGER.info("Loaded %s messages from %s", f"{len(df):,}", args.input)
    return df


def validate_final_corpus(df: pd.DataFrame, skip: bool) -> None:
    if skip:
        LOGGER.warning(
            "Final corpus validation was skipped. Do not use this option for "
            "the empirical publication run."
        )
        return
    observed_messages = df.groupby("_direction").size().to_dict()
    observed_units = df.groupby("_direction")["_unit_id"].nunique().to_dict()
    errors: list[str] = []
    for direction in ("direct", "group"):
        if observed_messages.get(direction) != EXPECTED_MESSAGES[direction]:
            errors.append(
                f"{direction} messages: observed "
                f"{observed_messages.get(direction)}, expected "
                f"{EXPECTED_MESSAGES[direction]}"
            )
        if observed_units.get(direction) != EXPECTED_UNITS[direction]:
            errors.append(
                f"{direction} units: observed {observed_units.get(direction)}, "
                f"expected {EXPECTED_UNITS[direction]}"
            )
    if errors:
        raise ValueError("Corpus validation failed: " + "; ".join(errors))


def unique_join(values: pd.Series) -> str:
    items = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return " | ".join(items)


def create_inventory(df: pd.DataFrame, args: argparse.Namespace) -> Path:
    group = df[df["_direction"].eq("group")].copy()
    for column in (args.team_col, args.channel_col):
        if column not in group.columns:
            group[column] = ""

    inventory = (
        group.groupby("_unit_id", as_index=False)
        .agg(
            team_name=(args.team_col, unique_join),
            channel_name=(args.channel_col, unique_join),
            messages=("_message_id", "size"),
            cleaned_tokens=("_token_count", "sum"),
        )
        .rename(columns={"_unit_id": "unit_id"})
        .sort_values(["team_name", "channel_name", "unit_id"], kind="stable")
        .reset_index(drop=True)
    )
    inventory["category"] = ""
    inventory.loc[inventory["team_name"].eq(""), "category"] = "unattributed"
    inventory["reviewer_notes"] = ""

    team_summary = (
        inventory.assign(
            team_name=inventory["team_name"].replace("", "[missing]")
        )
        .groupby("team_name", as_index=False)
        .agg(
            channels=("unit_id", "nunique"),
            messages=("messages", "sum"),
            cleaned_tokens=("cleaned_tokens", "sum"),
        )
        .sort_values("messages", ascending=False, kind="stable")
    )

    instructions = pd.DataFrame(
        {
            "instruction": [
                "Complete category for every unit_id.",
                "Allowed value: psychosomatic_core",
                "Allowed value: adjacent_clinical_team",
                "Allowed value: unattributed",
                "Do not publish this workbook; it contains confidential metadata.",
            ]
        }
    )

    args.confidential_dir.mkdir(parents=True, exist_ok=True)
    output = args.confidential_dir / "team_channel_classification.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        inventory.to_excel(writer, sheet_name="classification", index=False)
        team_summary.to_excel(writer, sheet_name="team_summary", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
    LOGGER.info("Confidential classification workbook written to %s", output)
    return output


def load_classification(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Classification file not found: {path}. Run --mode inventory first."
        )
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        classification = pd.read_excel(path, sheet_name="classification")
    else:
        classification = pd.read_csv(path)
    require_columns(classification, ("unit_id", "category"), "Classification")
    classification = classification.copy()
    classification["unit_id"] = clean_string(classification["unit_id"])
    classification["category"] = clean_string(
        classification["category"]
    ).str.casefold()
    if classification["unit_id"].duplicated().any():
        duplicates = classification.loc[
            classification["unit_id"].duplicated(keep=False), "unit_id"
        ].tolist()
        raise ValueError(
            "Classification contains duplicate unit_id values: "
            f"{duplicates[:10]}"
        )
    blanks = classification["category"].eq("")
    if blanks.any():
        raise ValueError(
            f"Classification is incomplete: {int(blanks.sum())} units have no category."
        )
    invalid = sorted(set(classification["category"]) - set(VALID_CATEGORIES))
    if invalid:
        raise ValueError(
            f"Invalid classification categories: {invalid}. "
            f"Allowed categories: {list(VALID_CATEGORIES)}"
        )
    return classification[["unit_id", "category"]]


def attach_classification(
    df: pd.DataFrame, classification: pd.DataFrame
) -> pd.DataFrame:
    group_units = set(df.loc[df["_direction"].eq("group"), "_unit_id"])
    classified_units = set(classification["unit_id"])
    missing = sorted(group_units - classified_units)
    extra = sorted(classified_units - group_units)
    if missing or extra:
        raise ValueError(
            "Classification does not match the final Group corpus. "
            f"Missing units={missing[:10]} (n={len(missing)}); "
            f"extra units={extra[:10]} (n={len(extra)})."
        )
    merged = df.merge(
        classification,
        how="left",
        left_on="_unit_id",
        right_on="unit_id",
        validate="many_to_one",
    )
    merged.loc[merged["_direction"].eq("direct"), "category"] = "direct"
    return merged.drop(columns=["unit_id"])


def build_corpora(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    direct = df[df["_direction"].eq("direct")].copy()
    group = df[df["_direction"].eq("group")].copy()
    corpora = {
        "direct": direct,
        "group_full": group,
        "group_nonadjacent": group[
            ~group["category"].eq("adjacent_clinical_team")
        ].copy(),
        "group_psychosomatic": group[
            group["category"].eq("psychosomatic_core")
        ].copy(),
    }
    if corpora["group_psychosomatic"].empty:
        raise ValueError("The psychosomatic-only Group corpus is empty.")
    if group["category"].eq("adjacent_clinical_team").sum() == 0:
        raise ValueError(
            "No messages were classified as adjacent_clinical_team. "
            "A sensitivity comparison cannot be performed."
        )
    return corpora


def summarize_composition(
    df: pd.DataFrame, corpora: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name, subset in corpora.items():
        rows.append(
            {
                "corpus": name,
                "communication_units": int(subset["_unit_id"].nunique()),
                "messages": int(len(subset)),
                "cleaned_tokens": int(subset["_token_count"].sum()),
                "nonempty_lexical_messages": int(
                    subset["_token_count"].gt(0).sum()
                ),
                "tokens_per_message": (
                    float(subset["_token_count"].sum() / len(subset))
                    if len(subset)
                    else math.nan
                ),
            }
        )
    category_rows = []
    group = df[df["_direction"].eq("group")]
    for category in VALID_CATEGORIES:
        subset = group[group["category"].eq(category)]
        category_rows.append(
            {
                "category": category,
                "channels": int(subset["_unit_id"].nunique()),
                "messages": int(len(subset)),
                "share_of_group_messages_pct": 100.0 * len(subset) / len(group),
                "cleaned_tokens": int(subset["_token_count"].sum()),
                "share_of_group_tokens_pct": (
                    100.0
                    * subset["_token_count"].sum()
                    / group["_token_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(category_rows)


def corpus_token_counts(df: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for tokens in df["_tokens"]:
        counter.update(tokens)
    return counter


def g2_log_likelihood(
    count_direct: int,
    count_group: int,
    total_direct: int,
    total_group: int,
) -> float:
    observed_total = count_direct + count_group
    if observed_total == 0 or total_direct + total_group == 0:
        return 0.0
    expected_direct = observed_total * total_direct / (total_direct + total_group)
    expected_group = observed_total * total_group / (total_direct + total_group)
    value = 0.0
    if count_direct > 0 and expected_direct > 0:
        value += count_direct * math.log(count_direct / expected_direct)
    if count_group > 0 and expected_group > 0:
        value += count_group * math.log(count_group / expected_group)
    return 2.0 * value


def unit_coverage(df: pd.DataFrame, item: str) -> tuple[int, float]:
    positive_units = int(
        df.assign(_present=df["_tokens"].map(lambda tokens: item in tokens))
        .groupby("_unit_id")["_present"]
        .any()
        .sum()
    )
    units = int(df["_unit_id"].nunique())
    return positive_units, 100.0 * positive_units / units if units else math.nan


def calculate_keyness(
    direct: pd.DataFrame,
    group: pd.DataFrame,
    comparison: str,
    min_total_frequency: int,
) -> pd.DataFrame:
    direct_counts = corpus_token_counts(direct)
    group_counts = corpus_token_counts(group)
    direct_total = sum(direct_counts.values())
    group_total = sum(group_counts.values())
    vocabulary = sorted(set(direct_counts) | set(group_counts))
    rows = []
    for item in vocabulary:
        d_count = int(direct_counts[item])
        g_count = int(group_counts[item])
        combined = d_count + g_count
        if combined < min_total_frequency:
            continue
        d_rate = 1000.0 * d_count / direct_total if direct_total else math.nan
        g_rate = 1000.0 * g_count / group_total if group_total else math.nan
        corrected_d_rate = (d_count + 0.5) / direct_total
        corrected_g_rate = (g_count + 0.5) / group_total
        log_ratio = math.log2(corrected_d_rate / corrected_g_rate)
        rows.append(
            {
                "comparison": comparison,
                "item": item,
                "direct_count": d_count,
                "group_count": g_count,
                "combined_count": combined,
                "direct_per_1000_tokens": d_rate,
                "group_per_1000_tokens": g_rate,
                "difference_direct_minus_group_per_1000": d_rate - g_rate,
                "log_ratio_direct_vs_group": log_ratio,
                "log_likelihood_g2": g2_log_likelihood(
                    d_count, g_count, direct_total, group_total
                ),
                "dominant_modality": "direct" if log_ratio > 0 else "group",
                "direct_tokens_denominator": direct_total,
                "group_tokens_denominator": group_total,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["log_likelihood_g2", "combined_count", "item"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def calculate_keyness_all(
    corpora: dict[str, pd.DataFrame],
    min_total_frequency: int,
    selected_items: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    full_tables: dict[str, pd.DataFrame] = {}
    selected_tables = []
    for group_name in (
        "group_full",
        "group_nonadjacent",
        "group_psychosomatic",
    ):
        table = calculate_keyness(
            corpora["direct"],
            corpora[group_name],
            comparison=f"direct_vs_{group_name}",
            min_total_frequency=min_total_frequency,
        )
        full_tables[group_name] = table
        selected = table[table["item"].isin(selected_items)].copy()
        if not selected.empty:
            direct_coverage = {
                item: unit_coverage(corpora["direct"], item)
                for item in selected["item"]
            }
            group_coverage = {
                item: unit_coverage(corpora[group_name], item)
                for item in selected["item"]
            }
            selected["direct_units_with_item"] = selected["item"].map(
                lambda item: direct_coverage[item][0]
            )
            selected["direct_unit_coverage_pct"] = selected["item"].map(
                lambda item: direct_coverage[item][1]
            )
            selected["group_units_with_item"] = selected["item"].map(
                lambda item: group_coverage[item][0]
            )
            selected["group_unit_coverage_pct"] = selected["item"].map(
                lambda item: group_coverage[item][1]
            )
        missing_items = sorted(set(selected_items) - set(selected["item"]))
        if missing_items:
            LOGGER.warning(
                "%s selected items were absent below the frequency threshold in %s: %s",
                len(missing_items),
                group_name,
                missing_items,
            )
        selected_tables.append(selected)

    selected_long = pd.concat(selected_tables, ignore_index=True)
    full_selected = selected_long[
        selected_long["comparison"].eq("direct_vs_group_full")
    ][["item", "group_per_1000_tokens"]].rename(
        columns={"group_per_1000_tokens": "group_full_per_1000_tokens"}
    )
    stability = selected_long.merge(full_selected, on="item", how="left")
    stability["group_rate_change_from_full_pct"] = stability.apply(
        lambda row: (
            100.0
            * (
                row["group_per_1000_tokens"]
                - row["group_full_per_1000_tokens"]
            )
            / row["group_full_per_1000_tokens"]
            if row["group_full_per_1000_tokens"] != 0
            else math.nan
        ),
        axis=1,
    )
    stability["direction_same_as_full"] = stability.groupby("item")[
        "dominant_modality"
    ].transform(lambda values: values.eq(values.iloc[0]))
    return full_tables, selected_long, stability


def top_share(values: pd.Series, top_n: int) -> float:
    positive = values[values > 0]
    total = positive.sum()
    if total == 0:
        return math.nan
    return 100.0 * positive.nlargest(top_n).sum() / total


def calculate_marker_summary(
    corpora: dict[str, pd.DataFrame], markers: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for corpus_name in (
        "group_full",
        "group_nonadjacent",
        "group_psychosomatic",
    ):
        subset = corpora[corpus_name]
        units_total = int(subset["_unit_id"].nunique())
        messages_total = int(len(subset))
        for marker in markers:
            message_presence = subset["_marker_tokens"].map(
                lambda tokens: marker in tokens
            )
            occurrences = subset["_marker_tokens"].map(
                lambda tokens: tokens.count(marker)
            )
            by_unit_messages = (
                pd.DataFrame(
                    {
                        "unit": subset["_unit_id"],
                        "present": message_presence.astype(int),
                    }
                )
                .groupby("unit")["present"]
                .sum()
            )
            messages_with = int(message_presence.sum())
            channels_with = int((by_unit_messages > 0).sum())
            rows.append(
                {
                    "corpus": corpus_name,
                    "marker": marker,
                    "group_channels_total": units_total,
                    "group_messages_total": messages_total,
                    "messages_with_marker": messages_with,
                    "message_coverage_pct": (
                        100.0 * messages_with / messages_total
                        if messages_total
                        else math.nan
                    ),
                    "marker_occurrences": int(occurrences.sum()),
                    "channels_with_marker": channels_with,
                    "channel_coverage_pct": (
                        100.0 * channels_with / units_total
                        if units_total
                        else math.nan
                    ),
                    "top1_channel_share_of_positive_messages_pct": top_share(
                        by_unit_messages, 1
                    ),
                    "top3_channel_share_of_positive_messages_pct": top_share(
                        by_unit_messages, 3
                    ),
                }
            )
    result = pd.DataFrame(rows)
    full_rates = result[result["corpus"].eq("group_full")][
        ["marker", "message_coverage_pct"]
    ].rename(columns={"message_coverage_pct": "group_full_message_coverage_pct"})
    result = result.merge(full_rates, on="marker", how="left")
    result["message_coverage_change_from_full_pct"] = result.apply(
        lambda row: (
            100.0
            * (
                row["message_coverage_pct"]
                - row["group_full_message_coverage_pct"]
            )
            / row["group_full_message_coverage_pct"]
            if row["group_full_message_coverage_pct"] != 0
            else math.nan
        ),
        axis=1,
    )
    return result


def validate_full_group_marker_counts(
    marker_summary: pd.DataFrame,
    markers: Sequence[str],
    skip: bool,
) -> None:
    if skip:
        LOGGER.warning(
            "Full-Group marker validation was skipped together with corpus "
            "validation. Do not use this option for the publication run."
        )
        return
    requested = set(markers)
    expected = {
        marker: count
        for marker, count in EXPECTED_FULL_GROUP_MARKER_MESSAGES.items()
        if marker in requested
    }
    full = marker_summary[marker_summary["corpus"].eq("group_full")]
    observed = dict(
        zip(full["marker"], full["messages_with_marker"], strict=False)
    )
    errors = []
    for marker, expected_count in expected.items():
        observed_count = observed.get(marker)
        if observed_count != expected_count:
            errors.append(
                f"{marker}: observed {observed_count}, expected {expected_count}"
            )
    if errors:
        raise ValueError(
            "Full-Group workflow-marker validation failed. Check "
            "--marker-text-col and marker tokenization: " + "; ".join(errors)
        )


def normalize_weekday(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return WEEKDAY_MAP.get(text.casefold())


def calculate_temporal_summary(
    corpora: dict[str, pd.DataFrame], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekday_rows = []
    hour_rows = []
    peak_rows = []
    for corpus_name in (
        "group_full",
        "group_nonadjacent",
        "group_psychosomatic",
    ):
        subset = corpora[corpus_name].copy()
        total = len(subset)
        if args.weekday_col in subset.columns:
            subset["_weekday_normalized"] = subset[args.weekday_col].map(
                normalize_weekday
            )
            unknown = int(subset["_weekday_normalized"].isna().sum())
            if unknown:
                LOGGER.warning(
                    "%s messages in %s have an unrecognized weekday.",
                    unknown,
                    corpus_name,
                )
            counts = subset["_weekday_normalized"].value_counts()
            for weekday in WEEKDAY_ORDER:
                count = int(counts.get(weekday, 0))
                weekday_rows.append(
                    {
                        "corpus": corpus_name,
                        "weekday": weekday,
                        "messages": count,
                        "total_messages": total,
                        "percentage": 100.0 * count / total if total else math.nan,
                    }
                )
            valid_counts = counts[counts.index.notna()]
            peak_weekday = valid_counts.idxmax() if not valid_counts.empty else None
            peak_weekday_messages = (
                int(valid_counts.max()) if not valid_counts.empty else 0
            )
        else:
            LOGGER.warning(
                "Weekday column %r is absent; weekday sensitivity is skipped.",
                args.weekday_col,
            )
            peak_weekday = None
            peak_weekday_messages = 0

        peak_hour = None
        peak_hour_messages = 0
        if args.timestamp_col in subset.columns:
            timestamps = pd.to_datetime(
                subset[args.timestamp_col],
                unit="s",
                errors="coerce",
            )
            parseable = int(timestamps.notna().sum())
            if parseable:
                hours = timestamps.dt.hour
                counts = hours.value_counts()
                for hour in range(24):
                    count = int(counts.get(hour, 0))
                    hour_rows.append(
                        {
                            "corpus": corpus_name,
                            "hour": hour,
                            "messages": count,
                            "parseable_messages": parseable,
                            "percentage": 100.0 * count / parseable,
                        }
                    )
                peak_hour = int(counts.idxmax())
                peak_hour_messages = int(counts.max())
            else:
                LOGGER.warning(
                    "No parseable timestamps in %s; hourly sensitivity is skipped.",
                    corpus_name,
                )
        else:
            LOGGER.warning(
                "Timestamp column %r is absent; hourly sensitivity is skipped.",
                args.timestamp_col,
            )
        peak_rows.append(
            {
                "corpus": corpus_name,
                "peak_weekday": peak_weekday,
                "peak_weekday_messages": peak_weekday_messages,
                "peak_weekday_pct": (
                    100.0 * peak_weekday_messages / total if total else math.nan
                ),
                "peak_hour": peak_hour,
                "peak_hour_messages": peak_hour_messages,
                "peak_hour_pct": (
                    100.0 * peak_hour_messages / total if total else math.nan
                ),
            }
        )
    return (
        pd.DataFrame(weekday_rows),
        pd.DataFrame(hour_rows),
        pd.DataFrame(peak_rows),
    )


def read_tabular(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def first_existing_column(
    df: pd.DataFrame, explicit: str | None, candidates: Sequence[str], label: str
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Medical {label} column not found: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def calculate_medical_sensitivity(
    df: pd.DataFrame,
    corpora: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.medical_mentions is None:
        return pd.DataFrame(), pd.DataFrame()
    if not args.medical_mentions.exists():
        raise FileNotFoundError(
            f"Medical-mentions file not found: {args.medical_mentions}"
        )
    mentions = read_tabular(args.medical_mentions, args.medical_sheet)
    message_col = first_existing_column(
        mentions,
        args.medical_message_id_col,
        ("message_id", "id", "utterance_id", "source_message_id"),
        "message identifier",
    )
    if message_col is None:
        raise ValueError(
            "Could not detect a message identifier in the medical-mentions file. "
            "Use --medical-message-id-col."
        )
    term_col = first_existing_column(
        mentions,
        args.medical_term_col,
        ("final_preferred_term", "preferred_term", "final_term", "term"),
        "term",
    )
    type_col = first_existing_column(
        mentions,
        args.medical_type_col,
        ("final_entity_type", "entity_type", "final_type", "type"),
        "entity type",
    )
    mentions = mentions.copy()
    mentions["_message_id"] = clean_string(mentions[message_col])
    mentions = mentions[mentions["_message_id"].ne("")].copy()
    unknown_ids = sorted(set(mentions["_message_id"]) - set(df["_message_id"]))
    if unknown_ids:
        LOGGER.warning(
            "%s medical rows refer to message IDs outside the corpus; they are excluded.",
            len(unknown_ids),
        )
        mentions = mentions[mentions["_message_id"].isin(set(df["_message_id"]))]

    dedup_columns = ["_message_id"]
    if term_col:
        mentions["_term"] = clean_string(mentions[term_col])
        dedup_columns.append("_term")
    if type_col:
        mentions["_type"] = clean_string(mentions[type_col])
        dedup_columns.append("_type")
    mentions = mentions.drop_duplicates(dedup_columns)

    summary_rows = []
    type_rows = []
    for corpus_name in (
        "group_full",
        "group_nonadjacent",
        "group_psychosomatic",
    ):
        subset = corpora[corpus_name]
        ids = set(subset["_message_id"])
        selected = mentions[mentions["_message_id"].isin(ids)]
        analyzable = subset[subset["_token_count"].gt(0)]
        tokens = int(analyzable["_token_count"].sum())
        entity_messages = int(selected["_message_id"].nunique())
        summary_rows.append(
            {
                "corpus": corpus_name,
                "validated_entity_mentions": int(len(selected)),
                "mentions_per_1000_cleaned_tokens": (
                    1000.0 * len(selected) / tokens if tokens else math.nan
                ),
                "messages_with_entity": entity_messages,
                "analyzable_messages": int(len(analyzable)),
                "message_coverage_pct": (
                    100.0 * entity_messages / len(analyzable)
                    if len(analyzable)
                    else math.nan
                ),
                "cleaned_tokens": tokens,
                "unique_terms": (
                    int(selected["_term"].nunique()) if term_col else math.nan
                ),
            }
        )
        if type_col:
            type_counts = selected["_type"].replace("", "[missing]").value_counts()
            for entity_type, count in type_counts.items():
                type_rows.append(
                    {
                        "corpus": corpus_name,
                        "entity_type": entity_type,
                        "mentions": int(count),
                        "share_of_mentions_pct": (
                            100.0 * count / len(selected)
                            if len(selected)
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(type_rows)


def calculate_confidential_unit_detail(
    df: pd.DataFrame, markers: Sequence[str]
) -> pd.DataFrame:
    group = df[df["_direction"].eq("group")].copy()
    rows = []
    for unit_id, subset in group.groupby("_unit_id", sort=True):
        row = {
            "unit_id": unit_id,
            "category": subset["category"].iloc[0],
            "messages": int(len(subset)),
            "cleaned_tokens": int(subset["_token_count"].sum()),
        }
        for marker in markers:
            presence = subset["_marker_tokens"].map(
                lambda tokens: marker in tokens
            )
            row[f"{marker}_messages"] = int(presence.sum())
            row[f"{marker}_occurrences"] = int(
                subset["_marker_tokens"]
                .map(lambda tokens: tokens.count(marker))
                .sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def round_numeric(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    result = df.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(decimals)
    return result


def write_public_outputs(
    args: argparse.Namespace,
    composition: pd.DataFrame,
    category_composition: pd.DataFrame,
    keyness_tables: dict[str, pd.DataFrame],
    selected_keyness: pd.DataFrame,
    keyness_stability: pd.DataFrame,
    marker_summary: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    temporal_peaks: pd.DataFrame,
    medical_summary: pd.DataFrame,
    medical_types: pd.DataFrame,
    config: RunConfig,
) -> None:
    args.public_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "team_composition_by_corpus.csv": composition,
        "team_composition_by_category.csv": category_composition,
        "keyness_sensitivity_selected_items.csv": selected_keyness,
        "keyness_sensitivity_stability.csv": keyness_stability,
        "workflow_marker_team_sensitivity.csv": marker_summary,
        "temporal_team_sensitivity_weekday.csv": weekday_summary,
        "temporal_team_sensitivity_hour.csv": hour_summary,
        "temporal_team_sensitivity_peaks.csv": temporal_peaks,
    }
    if not medical_summary.empty:
        outputs["medical_terminology_team_sensitivity.csv"] = medical_summary
    if not medical_types.empty:
        outputs["medical_terminology_team_sensitivity_by_type.csv"] = medical_types
    for filename, table in outputs.items():
        round_numeric(table).to_csv(
            args.public_dir / filename, index=False, encoding="utf-8-sig"
        )

    for corpus_name, table in keyness_tables.items():
        round_numeric(table).to_csv(
            args.public_dir / f"keyness_direct_vs_{corpus_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    workbook = args.public_dir / "team_sensitivity_summary.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        round_numeric(composition).to_excel(
            writer, sheet_name="corpus_composition", index=False
        )
        round_numeric(category_composition).to_excel(
            writer, sheet_name="team_categories", index=False
        )
        round_numeric(selected_keyness).to_excel(
            writer, sheet_name="selected_keyness", index=False
        )
        round_numeric(keyness_stability).to_excel(
            writer, sheet_name="keyness_stability", index=False
        )
        round_numeric(marker_summary).to_excel(
            writer, sheet_name="workflow_markers", index=False
        )
        round_numeric(weekday_summary).to_excel(
            writer, sheet_name="weekday", index=False
        )
        round_numeric(hour_summary).to_excel(
            writer, sheet_name="hour", index=False
        )
        round_numeric(temporal_peaks).to_excel(
            writer, sheet_name="temporal_peaks", index=False
        )
        if not medical_summary.empty:
            round_numeric(medical_summary).to_excel(
                writer, sheet_name="medical_summary", index=False
            )
        if not medical_types.empty:
            round_numeric(medical_types).to_excel(
                writer, sheet_name="medical_types", index=False
            )

    with (args.public_dir / "team_sensitivity_run_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_confidential_outputs(
    args: argparse.Namespace, unit_detail: pd.DataFrame
) -> None:
    args.confidential_dir.mkdir(parents=True, exist_ok=True)
    output = args.confidential_dir / "team_sensitivity_unit_detail.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        unit_detail.to_excel(writer, sheet_name="unit_detail", index=False)


def run_analysis(df: pd.DataFrame, args: argparse.Namespace) -> None:
    classification = load_classification(args.classification_file)
    classified = attach_classification(df, classification)
    corpora = build_corpora(classified)

    composition, category_composition = summarize_composition(classified, corpora)
    keyness_tables, selected_keyness, keyness_stability = calculate_keyness_all(
        corpora,
        min_total_frequency=args.min_total_frequency,
        selected_items=args.selected_key_items,
    )
    marker_summary = calculate_marker_summary(corpora, args.markers)
    validate_full_group_marker_counts(
        marker_summary, args.markers, args.skip_corpus_validation
    )
    weekday_summary, hour_summary, temporal_peaks = calculate_temporal_summary(
        corpora, args
    )
    medical_summary, medical_types = calculate_medical_sensitivity(
        classified, corpora, args
    )
    unit_detail = calculate_confidential_unit_detail(classified, args.markers)

    config = RunConfig(
        input_path=str(args.input),
        classification_file=str(args.classification_file),
        text_col=args.text_col,
        marker_text_col=args.marker_text_col,
        message_id_col=args.message_id_col,
        unit_col=args.unit_col,
        direction_col=args.direction_col,
        team_col=args.team_col,
        channel_col=args.channel_col,
        weekday_col=args.weekday_col,
        timestamp_col=args.timestamp_col,
        min_total_frequency=args.min_total_frequency,
        markers=list(args.markers),
        selected_key_items=list(args.selected_key_items),
        medical_mentions=(
            str(args.medical_mentions) if args.medical_mentions else None
        ),
    )

    write_public_outputs(
        args,
        composition,
        category_composition,
        keyness_tables,
        selected_keyness,
        keyness_stability,
        marker_summary,
        weekday_summary,
        hour_summary,
        temporal_peaks,
        medical_summary,
        medical_types,
        config,
    )
    write_confidential_outputs(args, unit_detail)

    LOGGER.info("Public outputs written to %s", args.public_dir)
    LOGGER.info("Confidential review output written to %s", args.confidential_dir)
    print("\nTeam-sensitivity corpus composition")
    print(round_numeric(composition, 3).to_string(index=False))
    print("\nWorkflow-marker sensitivity")
    display_columns = [
        "corpus",
        "marker",
        "messages_with_marker",
        "message_coverage_pct",
        "channels_with_marker",
        "channel_coverage_pct",
        "message_coverage_change_from_full_pct",
    ]
    print(round_numeric(marker_summary[display_columns], 3).to_string(index=False))
    print("\nAll requested team-sensitivity analyses completed successfully.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.min_total_frequency < 1:
        raise ValueError("--min-total-frequency must be at least 1.")
    df = load_corpus(args)
    validate_final_corpus(df, args.skip_corpus_validation)
    if args.mode == "inventory":
        output = create_inventory(df, args)
        print(f"Classification template created: {output}")
        print(
            "Complete the category column and then rerun the script with "
            "--mode analyze."
        )
    else:
        run_analysis(df, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise
