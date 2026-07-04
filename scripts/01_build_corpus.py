#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_corpus.py

Build raw analysis tables and optional ConvoKit corpus objects from exported
DocTalk chat CSV files.

Expected inputs
---------------
Direct message exports:
    data/direct/*.csv

Group message exports:
    data/group/*.csv

Optional:
    A manually curated group metadata validation file can be supplied via
    --manual_group_validation_file.

Confidential outputs
--------------------
Direct messages:
    outputs/confidential/dataframes/D_utterances_raw.csv
    outputs/confidential/dataframes/D_metadata_raw.csv
    outputs/confidential/corpus_objects/D_corpus/          (optional)

Group messages:
    outputs/confidential/dataframes/G_utterances_raw.csv
    outputs/confidential/dataframes/G_metadata_validated.csv
    outputs/confidential/review_files/metadata_validation/G_none_group_threads_for_manual_validation_with_full_texts_template.xlsx
    outputs/confidential/corpus_objects/G_corpus/          (optional)

Important notes
---------------
- Raw clinical chat exports should not be committed to a public repository.
- The group metadata validation template may contain original message text and
  is therefore written to outputs/confidential/review_files/.
- For GitHub/Zenodo, run the script on a synthetic sample or provide the code
  only, plus clear instructions for expected input structure.
- The timestamp conversion maps anonymized weekday + time strings to a fixed
  reference week (Monday 2022-01-03 to Sunday 2022-01-09). These timestamps are
  analytical placeholders and do not represent real dates.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


WEEKDAY_REFERENCE_DATES = {
    "Monday": "2022-01-03",
    "Tuesday": "2022-01-04",
    "Wednesday": "2022-01-05",
    "Thursday": "2022-01-06",
    "Friday": "2022-01-07",
    "Saturday": "2022-01-08",
    "Sunday": "2022-01-09",
}

DATE_PATTERN = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{2}:\d{2}$"
)


@dataclass
class BuildConfig:
    project_root: Path
    direct_dir: Path
    group_dir: Path
    dataframes_dir: Path
    corpus_objects_dir: Path
    results_table_dir: Path
    corpus: str
    encoding: str
    sep: str
    skipfooter: int
    usecols: Optional[List[int]]
    manual_group_validation_file: Optional[Path]
    write_excel: bool
    no_convokit: bool
    exclude_group_teams: List[str]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower()


def find_header_start(
    file_path: Path,
    *,
    encoding: str = "ISO-8859-1",
    max_lines: int = 10,
    fallback: int = 7,
) -> int:
    """
    Find the row where the message table starts.

    The Mattermost export files used in this project contain a metadata header,
    followed by a semicolon-separated message table. In the original notebooks,
    the table header was identified by searching for 'date' in the first lines.
    """
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if "date" in line.lower():
                return i
    return fallback


def read_metadata_from_csv(
    file_path: Path,
    *,
    encoding: str,
    sep: str,
    conversation_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Read the first six metadata rows of a Mattermost CSV export.

    The exported metadata have key-value structure in the first rows and are
    transposed to one metadata record per conversation.
    """
    conversation_id = conversation_id or file_path.stem

    try:
        meta_df = pd.read_csv(
            file_path,
            nrows=6,
            header=None,
            sep=sep,
            encoding=encoding,
            engine="python",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not read metadata from {file_path}: {exc}") from exc

    meta_df = meta_df.T
    meta_df.columns = meta_df.iloc[0]
    meta_df = meta_df.iloc[1:2].copy()
    meta_df = clean_column_names(meta_df)

    record = {
        "conversation_id": conversation_id,
        "Channel Name": None,
        "Channel Type": None,
        "Team Name": "none",
        "Anonymized": None,
        "Format": None,
    }

    for col in ["Channel Name", "Channel Type", "Team Name", "Anonymized"]:
        if col in meta_df.columns:
            record[col] = meta_df[col].iloc[0]

    format_col = next(
        (col for col in meta_df.columns if "Format" in str(col) or "Encoding" in str(col)),
        None,
    )
    if format_col:
        record["Format"] = meta_df[format_col].iloc[0]

    if pd.isna(record["Team Name"]) or str(record["Team Name"]).strip() == "":
        record["Team Name"] = "none"

    return record


def read_all_metadata(input_dir: Path, *, encoding: str, sep: str) -> pd.DataFrame:
    """
    Read metadata records from all CSV exports in a directory.

    Returns one metadata row per conversation/export file.
    """
    
    records: List[Dict[str, object]] = []

    for file_path in sorted(input_dir.glob("*.csv")):
        records.append(
            read_metadata_from_csv(
                file_path,
                encoding=encoding,
                sep=sep,
                conversation_id=file_path.stem,
            )
        )

    if not records:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    return pd.DataFrame(records)


def read_messages_from_csv(
    file_path: Path,
    *,
    encoding: str,
    sep: str,
    skipfooter: int,
    usecols: Optional[List[int]],
) -> pd.DataFrame:
    """
    Read the message table from one Mattermost CSV export.

    The table header is detected dynamically because exported files contain
    metadata rows before the actual message table.
    """
    header_start = find_header_start(file_path, encoding=encoding)

    read_kwargs = {
        "header": header_start,
        "skipfooter": skipfooter,
        "engine": "python",
        "encoding": encoding,
        "sep": sep,
    }
    if usecols is not None:
        read_kwargs["usecols"] = usecols

    df = pd.read_csv(file_path, **read_kwargs)
    df = clean_column_names(df)
    df["conversation_id"] = file_path.stem
    return df


def read_all_messages(
    input_dir: Path,
    *,
    encoding: str,
    sep: str,
    skipfooter: int,
    usecols: Optional[List[int]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for file_path in sorted(input_dir.glob("*.csv")):
        try:
            df = read_messages_from_csv(
                file_path,
                encoding=encoding,
                sep=sep,
                skipfooter=skipfooter,
                usecols=usecols,
            )
            frames.append(df)
        except Exception as exc:
            log(f"WARNING: Could not read message table from {file_path.name}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No readable message CSV files found in {input_dir}")

    return pd.concat(frames, ignore_index=True)


def validate_date_column(df: pd.DataFrame, *, label: str) -> None:
    """
    Validate that the message table contains anonymized weekday-time strings.

    Expected format is e.g. 'Monday 10:31'. Real calendar dates are not used.
    """
    if "date" not in df.columns:
        raise ValueError(
            f"Required column 'date' not found in {label}. "
            f"Available columns: {df.columns.tolist()}"
        )

    dates = df["date"].astype(str).str.strip()
    invalid_mask = ~dates.apply(lambda x: DATE_PATTERN.match(x) is not None)

    if invalid_mask.any():
        invalid_examples = df.loc[invalid_mask, ["conversation_id", "date"]].head(20)
        raise ValueError(
            f"Some date values in {label} do not match 'Weekday HH:MM'. "
            f"Examples:\n{invalid_examples.to_string(index=False)}"
        )


def add_analytical_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weekday, reference_date, datetime, timestamp and readable_date columns.

    The resulting timestamp is an analytical placeholder based on the fixed
    reference week. It is useful for time-of-week analyses and ConvoKit.
    """
    df = df.copy()
    df["date"] = df["date"].astype(str).str.strip()

    validate_date_column(df, label="message table")

    df["weekday"] = df["date"].str.split().str[0]
    df["time"] = df["date"].str.split().str[1]
    df["reference_date"] = df["weekday"].map(WEEKDAY_REFERENCE_DATES)

    if df["reference_date"].isna().any():
        missing = df.loc[df["reference_date"].isna(), "weekday"].unique()
        raise ValueError(f"Could not map weekday values: {missing}")

    df["datetime"] = pd.to_datetime(
        df["reference_date"] + " " + df["time"],
        format="%Y-%m-%d %H:%M",
        errors="raise",
    )

    df["timestamp"] = (
        (df["datetime"] - pd.Timestamp("1970-01-01"))
        // pd.Timedelta(seconds=1)
    ).astype("int64")

    df["readable_date"] = pd.to_datetime(df["timestamp"], unit="s")

    if not (df["datetime"] == df["readable_date"]).all():
        raise ValueError("Timestamp validation failed: datetime != readable_date.")

    return df


def standardize_message_columns(df: pd.DataFrame, *, corpus_prefix: str) -> pd.DataFrame:
    """
    Rename core export columns to ConvoKit-compatible column names and remove
    empty text messages.
    """
    df = clean_column_names(df)

    # Normalize likely column variants without destroying original metadata names.
    rename_map = {}
    lower_to_original = {normalize_column_name(c): c for c in df.columns}

    if "sender name" in lower_to_original:
        rename_map[lower_to_original["sender name"]] = "speaker"
    if "message" in lower_to_original:
        rename_map[lower_to_original["message"]] = "text"
    if "sender position" in lower_to_original:
        # Keep the downstream-compatible column with a space.
        rename_map[lower_to_original["sender position"]] = "sender position"

    df = df.rename(columns=rename_map)

    required = ["conversation_id", "speaker", "text", "timestamp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after standardization: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["reply_to"] = None

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    df = df.reset_index(drop=True)
    df["id"] = [f"{corpus_prefix}_{i}" for i in range(len(df))]

    # Ensure stable basic types.
    for col in ["id", "conversation_id", "speaker", "text"]:
        df[col] = df[col].astype(str)

    if "sender position" in df.columns:
        df["sender position"] = df["sender position"].fillna("").astype(str)

    df["timestamp"] = df["timestamp"].astype("int64")
    df["reply_to"] = None

    preferred = ["id", "conversation_id", "reply_to", "speaker", "timestamp", "text"]
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]


def drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove intermediate timestamp-construction columns before export.
    """
    helper_cols = [
        "date",
        "time",
        "reference_date",
        "calendar_date",
        "datetime",
        "readable_date",
    ]
    return df.drop(columns=[c for c in helper_cols if c in df.columns])


def normalize_team_name(name: object) -> str:
    if pd.isna(name):
        return "none"
    value = str(name).strip().lower()
    return value if value else "none"


def harmonize_team_name(name: str) -> str:
    """
    Collapse project-specific team-name variants to stable analysis labels.

    These rules reduce spelling/export variants while preserving analytically
    relevant team categories.
    """
    if name.startswith("endoskopie-"):
        return "endoskopie"
    if name.startswith("gastro-"):
        return "gastro"
    if name.startswith("na "):
        return "zna"
    if name.startswith("achi-"):
        return "achi"
    if name.startswith("inter-"):
        return "inter"
    return name


def find_column(df: pd.DataFrame, possible_names: Sequence[str]) -> Optional[str]:
    normalized = {normalize_column_name(col): col for col in df.columns}
    for name in possible_names:
        key = normalize_column_name(name)
        if key in normalized:
            return normalized[key]
    return None


def create_group_validation_template(
    metadata_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    output_file: Path,
) -> pd.DataFrame:
    """
    Create a confidential Excel template for manual validation of group threads
    with missing team metadata ('none').

    The template may include full thread text to support manual team assignment.
    It must therefore be written only to outputs/confidential/review_files/.
    This mirrors the group-message notebook validation logic.
    """
    none_threads = metadata_df[metadata_df["team_name_harmonized"] == "none"].copy()

    if none_threads.empty:
        template = none_threads.copy()
        template.to_excel(output_file, index=False, engine="openpyxl")
        return template

    message_col = find_column(messages_df, ["message", "text", "Textnachricht", "Nachricht"])
    date_col = find_column(messages_df, ["date", "Datum", "datetime", "timestamp"])
    sender_col = find_column(
        messages_df,
        ["sender name", "sender", "speaker", "sender position", "Sender"],
    )

    if message_col is None:
        raise ValueError(
            f"Could not find message column for validation template. "
            f"Available columns: {messages_df.columns.tolist()}"
        )

    text_df = messages_df.copy()
    text_df[message_col] = text_df[message_col].fillna("").astype(str).str.strip()
    text_df = text_df[text_df[message_col] != ""].copy()

    def build_line(row: pd.Series) -> str:
        parts: List[str] = []
        if date_col is not None and pd.notna(row.get(date_col)):
            parts.append(f"[{row[date_col]}]")
        if sender_col is not None and pd.notna(row.get(sender_col)):
            parts.append(f"{row[sender_col]}:")
        parts.append(str(row[message_col]))
        return " ".join(parts)

    text_df["thread_line"] = text_df.apply(build_line, axis=1)

    thread_texts = (
        text_df.groupby("conversation_id")
        .agg(
            number_of_utterances_loaded=("thread_line", "count"),
            full_thread_text=("thread_line", lambda x: "\n".join(x)),
        )
        .reset_index()
    )

    template = none_threads.merge(thread_texts, on="conversation_id", how="left")
    template["team_name_manual_validation"] = pd.NA
    template["manual_validation_notes"] = pd.NA

    keep = [
        "conversation_id",
        "Channel Name",
        "Channel Type",
        "Team Name",
        "team_name_normalized",
        "team_name_harmonized",
        "number_of_utterances_loaded",
        "full_thread_text",
        "team_name_manual_validation",
        "manual_validation_notes",
    ]
    keep = [c for c in keep if c in template.columns]
    template = template[keep].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    template.to_excel(output_file, index=False, engine="openpyxl")
    return template


def load_manual_group_validation(file_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if file_path is None:
        return None
    if not file_path.exists():
        log(f"WARNING: Manual group-validation file not found: {file_path}")
        return None

    df = pd.read_excel(file_path, engine="openpyxl")
    df = clean_column_names(df)

    if "conversation_id" not in df.columns:
        raise ValueError(
            f"Manual validation file has no 'conversation_id' column: {file_path}"
        )

    manual_col = "team_name_manual_validation"
    if manual_col not in df.columns:
        raise ValueError(
            f"Manual validation file has no '{manual_col}' column: {file_path}"
        )

    df = df[["conversation_id", manual_col]].copy()
    df["conversation_id"] = df["conversation_id"].astype(str).str.strip()
    df[manual_col] = (
        df[manual_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df.loc[df[manual_col] == "", manual_col] = pd.NA
    df = df.drop_duplicates(subset=["conversation_id"], keep="first")
    return df


def apply_group_team_validation(
    metadata_df: pd.DataFrame,
    manual_validation_df: Optional[pd.DataFrame],
    *,
    exclude_teams: Sequence[str],
) -> pd.DataFrame:
    df = metadata_df.copy()
    df["conversation_id"] = df["conversation_id"].astype(str).str.strip()

    df["team_name_normalized"] = df["Team Name"].apply(normalize_team_name)
    df["team_name_harmonized"] = df["team_name_normalized"].apply(harmonize_team_name)

    exclude = [normalize_team_name(x) for x in exclude_teams]
    before = len(df)
    df = df[~df["team_name_harmonized"].isin(exclude)].copy()
    log(f"Group metadata rows before exclusion: {before}")
    log(f"Group metadata rows after exclusion:  {len(df)}")
    log(f"Group metadata rows excluded:         {before - len(df)}")

    if manual_validation_df is not None:
        df = df.merge(manual_validation_df, on="conversation_id", how="left")
    else:
        df["team_name_manual_validation"] = pd.NA

    df["team_name_final"] = df["team_name_harmonized"]

    mask_manual = (
        (df["team_name_harmonized"] == "none")
        & df["team_name_manual_validation"].notna()
    )
    df.loc[mask_manual, "team_name_final"] = df.loc[
        mask_manual, "team_name_manual_validation"
    ]

    df["team_name_source"] = "metadata"
    df.loc[mask_manual, "team_name_source"] = "manual_validation"
    df.loc[df["team_name_final"] == "none", "team_name_source"] = "missing_metadata"

    return df


def clean_meta_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value


def row_value(row: pd.Series, column: str, default: object = None) -> object:
    if column in row.index:
        return clean_meta_value(row[column])
    return default


def build_convokit_corpus(
    df: pd.DataFrame,
    *,
    corpus_kind: str,
    output_path: Path,
) -> None:
    """
    Build and save a ConvoKit corpus object.

    For group messages, speaker labels are anonymized locally within each
    conversation. Therefore, speaker IDs are made conversation-specific to avoid
    merging different local '<Person1>' labels across group threads.
    """
    try:
        from convokit import Corpus, Speaker, Utterance
    except Exception as exc:
        log(f"WARNING: ConvoKit is not available. Skipping corpus dump. ({exc})")
        return

    df = df.copy()

    required = ["id", "speaker", "conversation_id", "reply_to", "timestamp", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build ConvoKit corpus. Missing columns: {missing}")

    speakers: Dict[str, object] = {}
    utterances: List[object] = []

    for _, row in df.iterrows():
        conversation_id = str(row["conversation_id"])
        local_speaker = str(row["speaker"])

        if corpus_kind == "group":
            speaker_id = f"{conversation_id}__{local_speaker}"
        else:
            speaker_id = local_speaker

        if speaker_id not in speakers:
            speakers[speaker_id] = Speaker(
                id=speaker_id,
                meta={
                    "local_speaker_label": local_speaker,
                    "sender_position": row_value(row, "sender position"),
                },
            )

        reply_to = row_value(row, "reply_to")
        if pd.isna(reply_to) or reply_to in ["", "None", "nan"]:
            reply_to = None

        if corpus_kind == "group":
            meta = {
                "channel_name": row_value(row, "Channel Name"),
                "channel_type": row_value(row, "Channel Type"),
                "team_name_original": row_value(row, "Team Name"),
                "team_name_normalized": row_value(row, "team_name_normalized"),
                "team_name_harmonized": row_value(row, "team_name_harmonized"),
                "team_name_manual_validation": row_value(row, "team_name_manual_validation"),
                "team_name_final": row_value(row, "team_name_final"),
                "team_name_source": row_value(row, "team_name_source"),
            }
        else:
            meta = {
                "channel_name": row_value(row, "Channel Name"),
                "channel_type": row_value(row, "Channel Type"),
                "team_name": row_value(row, "Team Name"),
            }

        utterances.append(
            Utterance(
                id=str(row["id"]),
                speaker=speakers[speaker_id],
                conversation_id=conversation_id,
                reply_to=reply_to,
                timestamp=int(row["timestamp"]),
                text=str(row["text"]),
                meta=meta,
            )
        )

    corpus = Corpus(utterances=utterances)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.dump(str(output_path))

    log(f"Saved ConvoKit corpus to: {output_path}")
    corpus.print_summary_stats()


def write_dataframe_outputs(
    df: pd.DataFrame,
    csv_path: Path,
    *,
    write_excel: bool,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log(f"Saved CSV: {csv_path}")

    if write_excel:
        xlsx_path = csv_path.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        log(f"Saved Excel: {xlsx_path}")


def build_direct(config: BuildConfig) -> Dict[str, object]:
    log("\n=== Building direct-message corpus ===")
    if not config.direct_dir.exists():
        raise FileNotFoundError(f"Direct input directory not found: {config.direct_dir}")

    metadata_df = read_all_metadata(
        config.direct_dir,
        encoding=config.encoding,
        sep=config.sep,
    )
    messages_df = read_all_messages(
        config.direct_dir,
        encoding=config.encoding,
        sep=config.sep,
        skipfooter=config.skipfooter,
        usecols=config.usecols,
    )

    messages_df = add_analytical_timestamps(messages_df)
    messages_df = standardize_message_columns(messages_df, corpus_prefix="D")
    messages_df = drop_helper_columns(messages_df)

    merged_df = messages_df.merge(metadata_df, on="conversation_id", how="left")

    empty_after_merge = merged_df["text"].isna().sum()
    if empty_after_merge:
        log(f"WARNING: Missing text values after merge: {empty_after_merge}")

    write_dataframe_outputs(
        metadata_df,
        config.dataframes_dir / "D_metadata_raw.csv",
        write_excel=config.write_excel,
    )
    write_dataframe_outputs(
        merged_df,
        config.dataframes_dir / "D_utterances_raw.csv",
        write_excel=config.write_excel,
    )

    if not config.no_convokit:
        build_convokit_corpus(
            merged_df,
            corpus_kind="direct",
            output_path=config.corpus_objects_dir / "D_corpus",
        )

    summary = {
        "corpus": "direct",
        "input_dir": str(config.direct_dir),
        "n_metadata_threads": int(metadata_df.shape[0]),
        "n_message_rows": int(merged_df.shape[0]),
        "n_unique_threads_with_messages": int(merged_df["conversation_id"].nunique()),
        "n_unique_speakers": int(merged_df["speaker"].nunique()),
        "output_csv": str(config.dataframes_dir / "D_utterances_raw.csv"),
    }

    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_group(config: BuildConfig) -> Dict[str, object]:
    log("\n=== Building group-message corpus ===")
    if not config.group_dir.exists():
        raise FileNotFoundError(f"Group input directory not found: {config.group_dir}")

    metadata_raw_df = read_all_metadata(
        config.group_dir,
        encoding=config.encoding,
        sep=config.sep,
    )
    messages_raw_df = read_all_messages(
        config.group_dir,
        encoding=config.encoding,
        sep=config.sep,
        skipfooter=config.skipfooter,
        usecols=config.usecols,
    )

    # Prepare metadata and optional manual validation.
    metadata_for_template = metadata_raw_df.copy()
    metadata_for_template["team_name_normalized"] = metadata_for_template["Team Name"].apply(normalize_team_name)
    metadata_for_template["team_name_harmonized"] = metadata_for_template["team_name_normalized"].apply(harmonize_team_name)

    template_file = (
        config.results_table_dir
        / "G_none_group_threads_for_manual_validation_with_full_texts_template.xlsx"
    )
    try:
        create_group_validation_template(
            metadata_for_template,
            messages_raw_df,
            template_file,
        )
        log(f"Saved/updated manual-validation template: {template_file}")
    except Exception as exc:
        log(f"WARNING: Could not create manual-validation template: {exc}")

    manual_df = load_manual_group_validation(config.manual_group_validation_file)
    metadata_validated_df = apply_group_team_validation(
        metadata_raw_df,
        manual_df,
        exclude_teams=config.exclude_group_teams,
    )

    messages_df = add_analytical_timestamps(messages_raw_df)
    messages_df = standardize_message_columns(messages_df, corpus_prefix="G")
    messages_df = drop_helper_columns(messages_df)

    included_ids = set(metadata_validated_df["conversation_id"].astype(str).str.strip())
    before_filter = len(messages_df)
    excluded_rows_df = messages_df[~messages_df["conversation_id"].isin(included_ids)].copy()
    messages_df = messages_df[messages_df["conversation_id"].isin(included_ids)].copy()
    log(f"Group message rows before metadata filter: {before_filter}")
    log(f"Group message rows after metadata filter:  {len(messages_df)}")
    log(f"Group message rows excluded:              {before_filter - len(messages_df)}")

    metadata_merge_cols = [
        "conversation_id",
        "Channel Name",
        "Channel Type",
        "Team Name",
        "Anonymized",
        "Format",
        "team_name_normalized",
        "team_name_harmonized",
        "team_name_manual_validation",
        "team_name_final",
        "team_name_source",
    ]
    metadata_merge_cols = [c for c in metadata_merge_cols if c in metadata_validated_df.columns]
    metadata_for_merge = metadata_validated_df[metadata_merge_cols].drop_duplicates(
        subset=["conversation_id"],
        keep="first",
    )

    merged_df = messages_df.merge(metadata_for_merge, on="conversation_id", how="left")

    missing_team = merged_df["team_name_final"].isna().sum() if "team_name_final" in merged_df.columns else len(merged_df)
    if missing_team:
        log(f"WARNING: Message rows without final team metadata: {missing_team}")
    else:
        log("Metadata merge validated: no missing team assignments.")

    metadata_thread_ids = set(metadata_validated_df["conversation_id"].astype(str).str.strip())
    message_thread_ids = set(merged_df["conversation_id"].astype(str).str.strip())
    metadata_without_messages = sorted(metadata_thread_ids - message_thread_ids)

    write_dataframe_outputs(
        metadata_validated_df,
        config.dataframes_dir / "G_metadata_validated.csv",
        write_excel=config.write_excel,
    )
    write_dataframe_outputs(
        merged_df,
        config.dataframes_dir / "G_utterances_raw.csv",
        write_excel=config.write_excel,
    )

    if not excluded_rows_df.empty:
        write_dataframe_outputs(
            excluded_rows_df,
            config.dataframes_dir / "G_excluded_message_rows.csv",
            write_excel=config.write_excel,
        )

    team_distribution = (
        metadata_validated_df["team_name_final"]
        .value_counts(dropna=False)
        .rename_axis("team_name_final")
        .reset_index(name="number_of_group_threads")
    )
    team_distribution["percentage_of_group_threads"] = (
        team_distribution["number_of_group_threads"]
        / team_distribution["number_of_group_threads"].sum()
        * 100
    ).round(2)
    write_dataframe_outputs(
        team_distribution,
        config.dataframes_dir / "G_group_message_threads_per_team_final_validated.csv",
        write_excel=config.write_excel,
    )

    if not config.no_convokit:
        build_convokit_corpus(
            merged_df,
            corpus_kind="group",
            output_path=config.corpus_objects_dir / "G_corpus",
        )

    summary = {
        "corpus": "group",
        "input_dir": str(config.group_dir),
        "n_metadata_threads_included": int(metadata_validated_df.shape[0]),
        "n_message_rows": int(merged_df.shape[0]),
        "n_unique_threads_with_messages": int(merged_df["conversation_id"].nunique()),
        "n_metadata_threads_without_message_rows": int(len(metadata_without_messages)),
        "metadata_threads_without_message_rows": metadata_without_messages,
        "n_unique_conversation_specific_speakers": int(
            (merged_df["conversation_id"].astype(str) + "__" + merged_df["speaker"].astype(str)).nunique()
        ),
        "output_csv": str(config.dataframes_dir / "G_utterances_raw.csv"),
    }

    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_usecols(value: str) -> Optional[List[int]]:
    if value.strip().lower() in {"", "none", "all"}:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip() != ""]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw DocTalk direct/group corpora from exported CSV files."
    )

    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("."),
        help="Project root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--corpus",
        choices=["direct", "group", "both"],
        default="both",
        help="Which corpus to build.",
    )
    parser.add_argument(
        "--direct_dir",
        type=Path,
        default=None,
        help="Directory with direct-message CSV exports. Default: <project_root>/data/direct",
    )
    parser.add_argument(
        "--group_dir",
        type=Path,
        default=None,
        help="Directory with group-message CSV exports. Default: <project_root>/data/group",
    )
    parser.add_argument(
        "--dataframes_dir",
        type=Path,
        default=None,
        help="Output directory for raw DataFrames. Default: <project_root>/outputs/confidential/dataframes",
    )
    parser.add_argument(
        "--corpus_objects_dir",
        type=Path,
        default=None,
        help="Output directory for ConvoKit corpus objects. Default: <project_root>/outputs/confidential/corpus_objects",
    )
    parser.add_argument(
        "--results_table_dir",
        type=Path,
        default=None,
        help="Output directory for confidential review templates. Default: <project_root>/outputs/confidential/review_files/metadata_validation",
    )
    parser.add_argument(
        "--manual_group_validation_file",
        type=Path,
        default=None,
        help=(
            "Optional Excel file with manual group team labels. "
            "Expected columns: conversation_id, team_name_manual_validation."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="ISO-8859-1",
        help="CSV input encoding. Default: ISO-8859-1.",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator. Default: ';'.",
    )
    parser.add_argument(
        "--skipfooter",
        type=int,
        default=5,
        help="Number of footer lines to skip in message CSV tables. Default: 5.",
    )
    parser.add_argument(
        "--usecols",
        default="0,1,2,3",
        help="Comma-separated zero-based column indices to read from message table, or 'all'. Default: 0,1,2,3.",
    )
    parser.add_argument(
        "--exclude_group_teams",
        default="doctalk-projekt,chatbotanfrage",
        help="Comma-separated harmonized group team names to exclude.",
    )
    parser.add_argument(
        "--write_excel",
        action="store_true",
        help="Also write Excel versions of output tables.",
    )
    parser.add_argument(
        "--no_convokit",
        action="store_true",
        help="Do not create ConvoKit corpus objects.",
    )

    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> BuildConfig:
    project_root = args.project_root.resolve()

    direct_dir = (args.direct_dir or (project_root / "data" / "direct")).resolve()
    group_dir = (args.group_dir or (project_root / "data" / "group")).resolve()

    dataframes_dir = (
        args.dataframes_dir
        or (project_root / "outputs" / "confidential" / "dataframes")
    ).resolve()
    corpus_objects_dir = (
        args.corpus_objects_dir
        or (project_root / "outputs" / "confidential" / "corpus_objects")
    ).resolve()
    results_table_dir = (
        args.results_table_dir
        or (
            project_root
            / "outputs"
            / "confidential"
            / "review_files"
            / "metadata_validation"
        )
    ).resolve()

    exclude_group_teams = [
        item.strip().lower()
        for item in args.exclude_group_teams.split(",")
        if item.strip()
    ]

    return BuildConfig(
        project_root=project_root,
        direct_dir=direct_dir,
        group_dir=group_dir,
        dataframes_dir=dataframes_dir,
        corpus_objects_dir=corpus_objects_dir,
        results_table_dir=results_table_dir,
        corpus=args.corpus,
        encoding=args.encoding,
        sep=args.sep,
        skipfooter=args.skipfooter,
        usecols=parse_usecols(args.usecols),
        manual_group_validation_file=args.manual_group_validation_file.resolve()
        if args.manual_group_validation_file is not None
        else None,
        write_excel=args.write_excel,
        no_convokit=args.no_convokit,
        exclude_group_teams=exclude_group_teams,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = make_config(args)

    ensure_dir(config.dataframes_dir)
    ensure_dir(config.corpus_objects_dir)
    ensure_dir(config.results_table_dir)

    run_summary: Dict[str, object] = {
        "script": "01_build_corpus.py",
        "project_root": str(config.project_root),
        "corpus_requested": config.corpus,
        "direct_dir": str(config.direct_dir),
        "group_dir": str(config.group_dir),
        "dataframes_dir": str(config.dataframes_dir),
        "corpus_objects_dir": str(config.corpus_objects_dir),
        "results_table_dir": str(config.results_table_dir),
        "encoding": config.encoding,
        "sep": config.sep,
        "skipfooter": config.skipfooter,
        "usecols": config.usecols,
        "manual_group_validation_file": str(config.manual_group_validation_file)
        if config.manual_group_validation_file
        else None,
        "exclude_group_teams": config.exclude_group_teams,
        "write_excel": config.write_excel,
        "no_convokit": config.no_convokit,
        "results": [],
    }

    if config.corpus in {"direct", "both"}:
        run_summary["results"].append(build_direct(config))

    if config.corpus in {"group", "both"}:
        run_summary["results"].append(build_group(config))

    summary_file = config.dataframes_dir / "01_build_corpus_run_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    log(f"\nSaved run summary: {summary_file}")
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
