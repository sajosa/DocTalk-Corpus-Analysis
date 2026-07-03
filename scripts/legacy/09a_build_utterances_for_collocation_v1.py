#!/usr/bin/env python3
"""
Build a combined utterance-level table for targeted collocation and KWIC analysis.

This version is adapted for post-cleaning utterance tables with:
    text_original
    text_clean_lexical

Default text source:
    text_clean_lexical

Usage from PROJECT ROOT:
    python scripts/09a_build_utterances_for_collocation.py \
        --direct outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv \
        --group outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv \
        --out outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical.csv \
        --marker-out outputs/results/collocations/marker_presence_check_clean_lexical.xlsx

Short usage if paths are unchanged:
    python scripts/09a_build_utterances_for_collocation.py

Optional: choose another text source column:
    python scripts/09a_build_utterances_for_collocation.py --text-col text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_MARKERS = [
    "PatName",
    "Hashtag_PatName",
    "KolName",
    "Mention_KolName",
    "Übergabe",
    "WE",
    "Todo",
    "kein_Todo",
    "kein",
    "Rückmeldung",
    "Gruppe",
    "MT",
    "GT",
    "KT",
    "Raum",
    "ÖGD",
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read a CSV file robustly by trying common encodings and automatic separator detection.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            print(f"Loaded {path.name} with encoding={encoding}")
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read CSV file: {path}\nLast error: {last_error}")


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    label: str,
) -> None:
    """
    Check that required columns exist.
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label}: missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize column names that differ between direct and group tables.
    Does not drop original information.
    """
    df = df.copy()

    # Harmonize sender position naming.
    if "sender_position" in df.columns and "sender position" not in df.columns:
        df["sender position"] = df["sender_position"]

    if "sender position" in df.columns and "sender_position" not in df.columns:
        df["sender_position"] = df["sender position"]

    # Add missing optional columns so concat and preferred ordering are stable.
    expected_optional_cols = [
        "reply_to",
        "weekday",
        "Channel Name",
        "Channel Type",
        "Team Name",
        "Anonymized",
        "Format",
        "text_original",
        "text_clean_lexical",
        "team_name_normalized",
        "team_name_harmonized",
        "team_name_manual_validation",
        "team_name_final",
        "team_name_source",
    ]

    for col in expected_optional_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def build_marker_presence_table(
    utterances: pd.DataFrame,
    text_col: str,
    markers: list[str],
) -> pd.DataFrame:
    """
    Build marker presence table using literal string matching.
    """
    text_series = utterances[text_col].fillna("").astype(str)
    rows = []

    for marker in markers:
        messages_with_marker = int(text_series.map(lambda text: marker in text).sum())
        total_occurrences = int(text_series.map(lambda text: text.count(marker)).sum())

        rows.append(
            {
                "marker": marker,
                "messages_with_marker": messages_with_marker,
                "total_occurrences": total_occurrences,
            }
        )

    return pd.DataFrame(rows)


def build_combined_utterance_table(
    direct_path: Path,
    group_path: Path,
    out_path: Path,
    marker_out_path: Path,
    text_source_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    required_base_cols = ["id", "conversation_id", "speaker", "timestamp"]
    required_text_cols = [text_source_col]

    print("\nLoading input files...")
    direct_df = read_csv_robust(direct_path)
    group_df = read_csv_robust(group_path)

    print(f"\nDirect shape: {direct_df.shape}")
    print(f"Group shape:  {group_df.shape}")

    print("\nDirect columns:")
    print(direct_df.columns.tolist())

    print("\nGroup columns:")
    print(group_df.columns.tolist())

    validate_required_columns(
        direct_df,
        required_base_cols + required_text_cols,
        "Direct table",
    )
    validate_required_columns(
        group_df,
        required_base_cols + required_text_cols,
        "Group table",
    )

    direct_df = harmonize_columns(direct_df)
    group_df = harmonize_columns(group_df)

    direct_df["direction"] = "direct"
    group_df["direction"] = "group"

    print(f"\nUsing text source column: {text_source_col}")

    # Main analysis columns for the collocation pipeline.
    direct_df["content_text"] = direct_df[text_source_col].fillna("").astype(str).str.strip()
    group_df["content_text"] = group_df[text_source_col].fillna("").astype(str).str.strip()

    # Placeholder until separate content/interaction views are created.
    # For now, both use the same cleaned lexical text.
    direct_df["interaction_text"] = direct_df["content_text"]
    group_df["interaction_text"] = group_df["content_text"]

    utterances = pd.concat([direct_df, group_df], ignore_index=True)

    before_empty_filter = len(utterances)
    utterances = utterances[utterances["content_text"] != ""].copy()
    after_empty_filter = len(utterances)

    print(f"\nRemoved empty messages: {before_empty_filter - after_empty_filter}")

    preferred_cols = [
        "direction",
        "id",
        "conversation_id",
        "reply_to",
        "speaker",
        "timestamp",
        "weekday",
        "sender position",
        "sender_position",
        "Channel Name",
        "Channel Type",
        "Team Name",
        "team_name_final",
        "team_name_source",
        "team_name_normalized",
        "team_name_harmonized",
        "team_name_manual_validation",
        "Anonymized",
        "Format",
        "text_original",
        "text",
        "text_clean_lexical",
        "content_text",
        "interaction_text",
    ]

    existing_cols = [col for col in preferred_cols if col in utterances.columns]
    remaining_cols = [col for col in utterances.columns if col not in existing_cols]
    utterances = utterances[existing_cols + remaining_cols].copy()

    print("\nCreating output folder if needed:")
    print(out_path.parent)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    utterances.to_csv(out_path, index=False, encoding="utf-8-sig")

    if not out_path.exists():
        raise RuntimeError(f"CSV output was not created: {out_path}")

    print("\nSaved combined utterance table:")
    print(out_path.resolve())
    print(f"File size: {out_path.stat().st_size:,} bytes")
    print(f"Combined shape: {utterances.shape}")

    print("\nDirection counts:")
    print(utterances["direction"].value_counts().to_string())

    marker_df = build_marker_presence_table(
        utterances=utterances,
        text_col="content_text",
        markers=DEFAULT_MARKERS,
    )

    print("\nCreating marker output folder if needed:")
    print(marker_out_path.parent)
    marker_out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if marker_out_path.suffix.lower() == ".csv":
            marker_df.to_csv(marker_out_path, index=False, encoding="utf-8-sig")
        else:
            marker_df.to_excel(marker_out_path, index=False)
    except Exception as exc:
        # Fallback if openpyxl is unavailable.
        fallback = marker_out_path.with_suffix(".csv")
        print(f"\nCould not write Excel marker file because: {exc}")
        print(f"Writing marker check as CSV instead: {fallback}")
        marker_df.to_csv(fallback, index=False, encoding="utf-8-sig")
        marker_out_path = fallback

    if not marker_out_path.exists():
        raise RuntimeError(f"Marker output was not created: {marker_out_path}")

    print("\nSaved marker presence check:")
    print(marker_out_path.resolve())
    print(f"File size: {marker_out_path.stat().st_size:,} bytes")

    print("\nMarker presence check:")
    print(marker_df.to_string(index=False))

    return utterances, marker_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined utterance table for targeted collocation analysis."
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory. Default: current working directory.",
    )

    parser.add_argument(
        "--direct",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv"),
        help="Path to direct utterance CSV relative to project dir or absolute.",
    )

    parser.add_argument(
        "--group",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv"),
        help="Path to group utterance CSV relative to project dir or absolute.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical.csv"),
        help="Output path for combined utterance CSV relative to project dir or absolute.",
    )

    parser.add_argument(
        "--marker-out",
        type=Path,
        default=Path("outputs/results/collocations/marker_presence_check_clean_lexical.xlsx"),
        help="Output path for marker presence check relative to project dir or absolute.",
    )

    parser.add_argument(
        "--text-col",
        type=str,
        default="text_clean_lexical",
        help="Column used as source for content_text and interaction_text. Default: text_clean_lexical.",
    )

    return parser.parse_args()


def resolve_path(path: Path, project_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return project_dir / path


def main() -> int:
    args = parse_args()

    project_dir = args.project_dir.resolve()

    direct_path = resolve_path(args.direct, project_dir)
    group_path = resolve_path(args.group, project_dir)
    out_path = resolve_path(args.out, project_dir)
    marker_out_path = resolve_path(args.marker_out, project_dir)

    print("Project directory:")
    print(project_dir)

    print("\nInput/output paths:")
    print(f"Direct:     {direct_path}")
    print(f"Group:      {group_path}")
    print(f"Combined:   {out_path}")
    print(f"Marker out: {marker_out_path}")
    print(f"Text col:   {args.text_col}")

    try:
        build_combined_utterance_table(
            direct_path=direct_path,
            group_path=group_path,
            out_path=out_path,
            marker_out_path=marker_out_path,
            text_source_col=args.text_col,
        )
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
