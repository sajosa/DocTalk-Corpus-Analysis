#!/usr/bin/env python3
"""
Generate a fully synthetic demonstration corpus matching the input schema
of the Code_KorpusAnalyse repository.

The generated messages are fictional, template-based, and not derived from
the confidential DocTalk corpus.

No real patient, employee, clinical case, conversation, or event is represented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_WEEK = pd.Timestamp("2022-01-03 00:00:00")


def make_datetime(day_offset: int, hour: int, minute: int = 0) -> pd.Timestamp:
    """Create a datetime within a fixed synthetic reference week."""
    return BASE_WEEK + pd.Timedelta(
        days=day_offset,
        hours=hour,
        minutes=minute,
    )


def to_unix_timestamp(value: pd.Timestamp) -> int:
    """Convert a pandas timestamp to Unix seconds."""
    return int(value.timestamp())


def build_direct_utterances() -> pd.DataFrame:
    """Create synthetic direct-message utterances."""

    messages = [
        (
            "D_SYN_001",
            "D_SYN_001_U001",
            None,
            "Person_D01",
            make_datetime(0, 8, 15),
            "Guten Morgen KolName, hast du heute kurz Zeit für eine Rückmeldung?",
            1,
        ),
        (
            "D_SYN_001",
            "D_SYN_001_U002",
            "D_SYN_001_U001",
            "Person_D02",
            make_datetime(0, 8, 18),
            "Ja, gerne. Ich melde mich nach der Besprechung.",
            2,
        ),
        (
            "D_SYN_001",
            "D_SYN_001_U003",
            "D_SYN_001_U002",
            "Person_D01",
            make_datetime(0, 10, 5),
            "Danke! Es geht um PatName und die Planung für morgen.",
            1,
        ),
        (
            "D_SYN_002",
            "D_SYN_002_U001",
            None,
            "Person_D03",
            make_datetime(1, 9, 10),
            "Ist Raum 3 von 10 bis 11 Uhr frei?",
            1,
        ),
        (
            "D_SYN_002",
            "D_SYN_002_U002",
            "D_SYN_002_U001",
            "Person_D04",
            make_datetime(1, 9, 14),
            "Raum 3 ist belegt, aber Raum 5 wäre verfügbar.",
            2,
        ),
        (
            "D_SYN_002",
            "D_SYN_002_U003",
            "D_SYN_002_U002",
            "Person_D03",
            make_datetime(1, 9, 16),
            "Super, vielen Dank 🙂",
            1,
        ),
        (
            "D_SYN_003",
            "D_SYN_003_U001",
            None,
            "Person_D05",
            make_datetime(2, 11, 30),
            "Kannst du bitte prüfen, ob die ÖGD für PatName schon terminiert ist?",
            1,
        ),
        (
            "D_SYN_003",
            "D_SYN_003_U002",
            "D_SYN_003_U001",
            "Person_D06",
            make_datetime(2, 11, 42),
            "Die ÖGD ist für morgen eingetragen.",
            2,
        ),
        (
            "D_SYN_003",
            "D_SYN_003_U003",
            "D_SYN_003_U002",
            "Person_D05",
            make_datetime(2, 11, 45),
            "Danke für die schnelle Rückmeldung.",
            1,
        ),
        (
            "D_SYN_004",
            "D_SYN_004_U001",
            None,
            "Person_D07",
            make_datetime(4, 13, 5),
            "Gibt es für Hashtag_PatName noch ein Todo für das WE?",
            1,
        ),
        (
            "D_SYN_004",
            "D_SYN_004_U002",
            "D_SYN_004_U001",
            "Person_D08",
            make_datetime(4, 13, 12),
            "Nein, aktuell kein aktives Todo.",
            2,
        ),
        (
            "D_SYN_004",
            "D_SYN_004_U003",
            "D_SYN_004_U002",
            "Person_D07",
            make_datetime(4, 13, 14),
            "Alles klar, danke 👍",
            1,
        ),
        (
            "D_SYN_005",
            "D_SYN_005_U001",
            None,
            "Person_D09",
            make_datetime(3, 14, 20),
            "Hallo Mention_KolName, kannst du die KT Gruppe morgen übernehmen?",
            1,
        ),
        (
            "D_SYN_005",
            "D_SYN_005_U002",
            "D_SYN_005_U001",
            "Person_D10",
            make_datetime(3, 14, 24),
            "Ja, ich übernehme die KT Gruppe gerne.",
            2,
        ),
    ]

    rows = []

    for (
        conversation_id,
        message_id,
        reply_to,
        speaker,
        dt,
        text,
        sender_position,
    ) in messages:
        rows.append(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "reply_to": reply_to,
                "speaker": speaker,
                "timestamp": to_unix_timestamp(dt),
                "text": text,
                "sender position": sender_position,
                "weekday": dt.weekday(),
                "Channel Name": conversation_id,
                "Channel Type": "direct",
                "Team Name": "synthetic_direct",
                "Anonymized": True,
                "Format": "synthetic",
            }
        )

    return pd.DataFrame(rows)


def build_direct_metadata() -> pd.DataFrame:
    """Create metadata matching the direct-message metadata schema."""

    rows = []

    for number in range(1, 6):
        conversation_id = f"D_SYN_{number:03d}"

        rows.append(
            {
                "conversation_id": conversation_id,
                "Channel Name": conversation_id,
                "Channel Type": "direct",
                "Team Name": "synthetic_direct",
                "Anonymized": True,
                "Format": "synthetic",
            }
        )

    return pd.DataFrame(rows)


def build_group_utterances() -> pd.DataFrame:
    """Create synthetic group-message utterances."""

    messages = [
        (
            "G_SYN_001",
            "G_SYN_001_U001",
            None,
            "Person_G01",
            make_datetime(4, 14, 0),
            "Übergabe für das WE:",
            1,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_001",
            "G_SYN_001_U002",
            "G_SYN_001_U001",
            "Person_G01",
            make_datetime(4, 14, 1),
            "PatName: kein Todo.",
            1,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_001",
            "G_SYN_001_U003",
            "G_SYN_001_U002",
            "Person_G01",
            make_datetime(4, 14, 2),
            "Hashtag_PatName: bitte morgen kurze Rückmeldung geben.",
            1,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_001",
            "G_SYN_001_U004",
            "G_SYN_001_U003",
            "Person_G02",
            make_datetime(4, 14, 10),
            "Danke für die Übergabe.",
            2,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_002",
            "G_SYN_002_U001",
            None,
            "Person_G03",
            make_datetime(1, 15, 0),
            "Rückmeldung aus der MT Gruppe: Alle waren anwesend.",
            1,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_002",
            "G_SYN_002_U002",
            "G_SYN_002_U001",
            "Person_G04",
            make_datetime(1, 15, 3),
            "PatName beteiligte sich aktiv an der MT Gruppe.",
            2,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_002",
            "G_SYN_002_U003",
            "G_SYN_002_U002",
            "Person_G05",
            make_datetime(1, 15, 8),
            "Vielen Dank für die Rückmeldung 🙂",
            3,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_003",
            "G_SYN_003_U001",
            None,
            "Person_G06",
            make_datetime(2, 13, 40),
            "Die GT Gruppe beginnt heute um 14 Uhr.",
            1,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_003",
            "G_SYN_003_U002",
            "G_SYN_003_U001",
            "Person_G07",
            make_datetime(2, 13, 43),
            "Bitte begleitet die Gruppe zum Raum 5.",
            2,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_003",
            "G_SYN_003_U003",
            "G_SYN_003_U002",
            "Person_G08",
            make_datetime(2, 13, 48),
            "KolName übernimmt die GT Gruppe.",
            3,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_004",
            "G_SYN_004_U001",
            None,
            "Person_G09",
            make_datetime(3, 10, 15),
            "Die ÖGD für PatName wurde auf morgen verschoben.",
            1,
            "Endoskopie",
            "endoskopie",
        ),
        (
            "G_SYN_004",
            "G_SYN_004_U002",
            "G_SYN_004_U001",
            "Person_G10",
            make_datetime(3, 10, 20),
            "Bitte nach dem Termin eine Rückmeldung senden.",
            2,
            "Endoskopie",
            "endoskopie",
        ),
        (
            "G_SYN_004",
            "G_SYN_004_U003",
            "G_SYN_004_U002",
            "Person_G11",
            make_datetime(3, 10, 23),
            "Wird gemacht.",
            3,
            "Endoskopie",
            "endoskopie",
        ),
        (
            "G_SYN_005",
            "G_SYN_005_U001",
            None,
            "Person_G12",
            make_datetime(0, 12, 0),
            "Wer kann morgen die KT Gruppe vertreten?",
            1,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_005",
            "G_SYN_005_U002",
            "G_SYN_005_U001",
            "Person_G13",
            make_datetime(0, 12, 5),
            "Ich kann die KT Gruppe übernehmen.",
            2,
            "Psychosomatik",
            "psom",
        ),
        (
            "G_SYN_005",
            "G_SYN_005_U003",
            "G_SYN_005_U002",
            "Person_G12",
            make_datetime(0, 12, 8),
            "Perfekt, danke!",
            1,
            "Psychosomatik",
            "psom",
        ),
    ]

    rows = []

    for (
        conversation_id,
        message_id,
        reply_to,
        speaker,
        dt,
        text,
        sender_position,
        team_name,
        harmonized_team,
    ) in messages:
        normalized_team = team_name.strip().lower()

        rows.append(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "reply_to": reply_to,
                "speaker": speaker,
                "timestamp": to_unix_timestamp(dt),
                "text": text,
                "sender position": sender_position,
                "weekday": dt.weekday(),
                "Channel Name": conversation_id,
                "Channel Type": "group",
                "Team Name": team_name,
                "Anonymized": True,
                "Format": "synthetic",
                "team_name_normalized": normalized_team,
                "team_name_harmonized": harmonized_team,
                "team_name_manual_validation": pd.NA,
                "team_name_final": harmonized_team,
                "team_name_source": "harmonized",
            }
        )

    return pd.DataFrame(rows)


def build_group_metadata() -> pd.DataFrame:
    """Create metadata matching the validated group metadata schema."""

    conversations = [
        ("G_SYN_001", "Psychosomatik", "psom"),
        ("G_SYN_002", "Psychosomatik", "psom"),
        ("G_SYN_003", "Psychosomatik", "psom"),
        ("G_SYN_004", "Endoskopie", "endoskopie"),
        ("G_SYN_005", "Psychosomatik", "psom"),
    ]

    rows = []

    for conversation_id, team_name, harmonized_team in conversations:
        rows.append(
            {
                "conversation_id": conversation_id,
                "Channel Name": conversation_id,
                "Channel Type": "group",
                "Team Name": team_name,
                "Anonymized": True,
                "Format": "synthetic",
                "team_name_normalized": team_name.strip().lower(),
                "team_name_harmonized": harmonized_team,
                "team_name_manual_validation": pd.NA,
                "team_name_final": harmonized_team,
                "team_name_source": "harmonized",
            }
        )

    return pd.DataFrame(rows)


def validate_columns(
    dataframe: pd.DataFrame,
    expected_columns: list[str],
    dataframe_name: str,
) -> None:
    """Validate exact column names and order."""

    actual_columns = dataframe.columns.tolist()

    if actual_columns != expected_columns:
        raise ValueError(
            f"Unexpected schema for {dataframe_name}.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {actual_columns}"
        )


def validate_synthetic_data(
    direct_utterances: pd.DataFrame,
    direct_metadata: pd.DataFrame,
    group_utterances: pd.DataFrame,
    group_metadata: pd.DataFrame,
) -> None:
    """Run integrity and schema validation."""

    direct_utterance_columns = [
        "id",
        "conversation_id",
        "reply_to",
        "speaker",
        "timestamp",
        "text",
        "sender position",
        "weekday",
        "Channel Name",
        "Channel Type",
        "Team Name",
        "Anonymized",
        "Format",
    ]

    direct_metadata_columns = [
        "conversation_id",
        "Channel Name",
        "Channel Type",
        "Team Name",
        "Anonymized",
        "Format",
    ]

    group_utterance_columns = [
        "id",
        "conversation_id",
        "reply_to",
        "speaker",
        "timestamp",
        "text",
        "sender position",
        "weekday",
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

    group_metadata_columns = [
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

    validate_columns(
        direct_utterances,
        direct_utterance_columns,
        "direct utterances",
    )
    validate_columns(
        direct_metadata,
        direct_metadata_columns,
        "direct metadata",
    )
    validate_columns(
        group_utterances,
        group_utterance_columns,
        "group utterances",
    )
    validate_columns(
        group_metadata,
        group_metadata_columns,
        "group metadata",
    )

    for name, dataframe in {
        "direct utterances": direct_utterances,
        "group utterances": group_utterances,
    }.items():
        if dataframe["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate message IDs.")

        if dataframe["text"].isna().any():
            raise ValueError(f"{name} contains missing message text.")

        if dataframe["text"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains empty message text.")

        if not dataframe["weekday"].between(0, 6).all():
            raise ValueError(f"{name} contains invalid weekday values.")

        if not pd.api.types.is_integer_dtype(dataframe["timestamp"]):
            raise ValueError(
                f"{name} timestamps are not stored as Unix integers."
            )

    if set(direct_utterances["conversation_id"]) != set(
        direct_metadata["conversation_id"]
    ):
        raise ValueError(
            "Direct-message conversation IDs do not match metadata."
        )

    if set(group_utterances["conversation_id"]) != set(
        group_metadata["conversation_id"]
    ):
        raise ValueError(
            "Group-message conversation IDs do not match metadata."
        )

    combined_text = " ".join(
        pd.concat(
            [
                direct_utterances["text"],
                group_utterances["text"],
            ],
            ignore_index=True,
        )
    )

    expected_markers = [
        "PatName",
        "Hashtag_PatName",
        "KolName",
        "Mention_KolName",
        "Übergabe",
        "WE",
        "Todo",
        "kein Todo",
        "kein aktives Todo",
        "Rückmeldung",
        "MT Gruppe",
        "GT Gruppe",
        "KT Gruppe",
        "Raum",
        "ÖGD",
    ]

    missing_markers = [
        marker
        for marker in expected_markers
        if marker not in combined_text
    ]

    if missing_markers:
        raise ValueError(
            "Synthetic corpus is missing expected markers: "
            f"{missing_markers}"
        )


def write_files(output_dir: Path) -> None:
    """Generate, validate, and save all synthetic input files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    direct_utterances = build_direct_utterances()
    direct_metadata = build_direct_metadata()
    group_utterances = build_group_utterances()
    group_metadata = build_group_metadata()

    validate_synthetic_data(
        direct_utterances=direct_utterances,
        direct_metadata=direct_metadata,
        group_utterances=group_utterances,
        group_metadata=group_metadata,
    )

    output_files = {
        "D_utterances_raw.csv": direct_utterances,
        "D_metadata_raw.csv": direct_metadata,
        "G_utterances_raw.csv": group_utterances,
        "G_metadata_validated.csv": group_metadata,
    }

    for filename, dataframe in output_files.items():
        path = output_dir / filename
        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"Created: {path} "
            f"({len(dataframe)} rows, "
            f"{len(dataframe.columns)} columns)"
        )

    print("\nSynthetic sample generated and validated successfully.")
    print("All messages and metadata are fictional.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fully synthetic corpus matching the repository's "
            "confidential input-table schemas."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_sample"),
        help=(
            "Directory for generated CSV files "
            "(default: data/synthetic_sample)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_files(args.output_dir)


if __name__ == "__main__":
    main()