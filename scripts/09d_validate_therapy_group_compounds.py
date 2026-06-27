#!/usr/bin/env python3
"""
Validate therapy group multiword expressions before v2 cleaning.

Checks occurrences of:
    MT Gruppe / Gruppe MT
    GT Gruppe / Gruppe GT
    KT Gruppe / Gruppe KT

Input:
    outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical.csv

Output:
    outputs/results/collocations/therapy_group_compound_validation.xlsx

Usage from project root:
    python scripts/09d_validate_therapy_group_compounds.py
"""

from pathlib import Path
import re
import pandas as pd


PROJECT_DIR = Path.cwd()

IN_PATH = PROJECT_DIR / "outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical.csv"
OUT_PATH = PROJECT_DIR / "outputs/results/collocations/therapy_group_compound_validation.xlsx"

TEXT_COL = "text_clean_lexical"

PATTERNS = {
    "MT_Gruppe_forward": r"\bMT\s+Gruppe\b",
    "MT_Gruppe_reverse": r"\bGruppe\s+MT\b",
    "GT_Gruppe_forward": r"\bGT\s+Gruppe\b",
    "GT_Gruppe_reverse": r"\bGruppe\s+GT\b",
    "KT_Gruppe_forward": r"\bKT\s+Gruppe\b",
    "KT_Gruppe_reverse": r"\bGruppe\s+KT\b",
    "MT_Gruppe_hyphen": r"\bMT[-_]?Gruppe\b",
    "GT_Gruppe_hyphen": r"\bGT[-_]?Gruppe\b",
    "KT_Gruppe_hyphen": r"\bKT[-_]?Gruppe\b",
}


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def count_pattern(text, pattern):
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def extract_kwic(text, pattern, window_chars=80, max_examples=5):
    examples = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(0, match.start() - window_chars)
        end = min(len(text), match.end() + window_chars)
        examples.append(
            {
                "left_context": text[start:match.start()],
                "match": text[match.start():match.end()],
                "right_context": text[match.end():end],
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    print("Loaded:", IN_PATH)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Available columns: {df.columns.tolist()}")

    if "direction" not in df.columns:
        df["direction"] = "unknown"

    team_col = None
    for candidate in ["team_name_final", "Team Name", "team_name_harmonized", "team_name_normalized"]:
        if candidate in df.columns:
            team_col = candidate
            break

    if team_col is None:
        df["team_context"] = "unknown"
        team_col = "team_context"

    rows = []
    kwic_rows = []

    for pattern_name, pattern in PATTERNS.items():
        compound = pattern_name.replace("_forward", "").replace("_reverse", "").replace("_hyphen", "")

        for idx, row in df.iterrows():
            text = safe_text(row[TEXT_COL])
            n = count_pattern(text, pattern)

            if n > 0:
                rows.append(
                    {
                        "pattern_name": pattern_name,
                        "compound_target": compound,
                        "direction": row.get("direction", "unknown"),
                        "team_context": row.get(team_col, "unknown"),
                        "conversation_id": row.get("conversation_id", ""),
                        "message_id": row.get("id", ""),
                        "occurrences_in_message": n,
                        "text_clean_lexical": text,
                    }
                )

                for ex in extract_kwic(text, pattern):
                    kwic_rows.append(
                        {
                            "pattern_name": pattern_name,
                            "compound_target": compound,
                            "direction": row.get("direction", "unknown"),
                            "team_context": row.get(team_col, "unknown"),
                            "conversation_id": row.get("conversation_id", ""),
                            "message_id": row.get("id", ""),
                            "left_context": ex["left_context"],
                            "match": ex["match"],
                            "right_context": ex["right_context"],
                        }
                    )

    matches = pd.DataFrame(rows)
    kwic = pd.DataFrame(kwic_rows)

    if matches.empty:
        summary = pd.DataFrame(
            columns=[
                "pattern_name",
                "compound_target",
                "direction",
                "team_context",
                "messages_with_pattern",
                "total_occurrences",
            ]
        )
    else:
        summary = (
            matches
            .groupby(["pattern_name", "compound_target", "direction", "team_context"], dropna=False)
            .agg(
                messages_with_pattern=("message_id", "nunique"),
                total_occurrences=("occurrences_in_message", "sum"),
            )
            .reset_index()
            .sort_values(
                by=["compound_target", "direction", "team_context", "total_occurrences"],
                ascending=[True, True, True, False],
            )
        )

    overall = (
        summary
        .groupby(["compound_target", "pattern_name"], dropna=False)
        .agg(
            messages_with_pattern=("messages_with_pattern", "sum"),
            total_occurrences=("total_occurrences", "sum"),
        )
        .reset_index()
        .sort_values(["compound_target", "total_occurrences"], ascending=[True, False])
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall_counts", index=False)
        summary.to_excel(writer, sheet_name="by_direction_team", index=False)
        matches.to_excel(writer, sheet_name="matched_messages", index=False)
        kwic.to_excel(writer, sheet_name="kwic_examples", index=False)

    print("Saved:", OUT_PATH)
    print("Overall counts:")
    print(overall)


if __name__ == "__main__":
    main()