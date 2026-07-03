#!/usr/bin/env python3
"""
14_emoji_analysis.py

Reproducible descriptive emoji-shortcode analysis for the DocTalk corpus.

The script recreates the notebook-based emoji analysis using the cleaned
utterance CSV files. Public outputs contain only aggregated counts, categories,
and shortcode codebooks. Message-level examples are not written unless the
explicit --write-confidential-examples flag is used.

Default inputs:
    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Default public outputs:
    outputs/public/tables/emoji/emoji_analysis_tables.xlsx
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import emoji as emoji_lib
except ImportError:  # pragma: no cover
    emoji_lib = None


MESSAGE_LABELS = {
    "direct": "Direct messages",
    "group": "Group messages",
}

SHORTCODE_PATTERN = re.compile(r":[A-Za-z0-9_+\-]+:")
NON_EMOJI_TIME_PATTERN = re.compile(r"^:[0-9]+(?:-[0-9]+)+:$")
NUMERIC_SHORTCODE_PATTERN = re.compile(r"^:[0-9]+:$")
EXCLUDED_UNICODE_SYMBOLS = {"®", "©", "™"}

NON_EMOJI_SHORTCODE_VALUES = {
    ":Patient:",
    ":innen:",
    ":1:",
    ":00:",
    ":20:",
}

EMOJI_CATEGORY_MAPPING = {
    # Approval / agreement
    ":+1:": "approval / agreement",
    ":ok_hand:": "approval / agreement",
    ":clap:": "approval / agreement",
    ":raised_hands:": "approval / agreement",
    ":white_check_mark:": "approval / agreement",
    ":heavy_check_mark:": "approval / agreement",
    ":v:": "approval / agreement",
    ":muscle:": "approval / agreement",
    ":rocket:": "approval / agreement",

    # Positive affect / friendliness
    ":blush:": "positive affect / friendliness",
    ":relaxed:": "positive affect / friendliness",
    ":slightly_smiling_face:": "positive affect / friendliness",
    ":smiley:": "positive affect / friendliness",
    ":smile:": "positive affect / friendliness",
    ":grinning:": "positive affect / friendliness",
    ":grin:": "positive affect / friendliness",
    ":wink:": "positive affect / friendliness",
    ":innocent:": "positive affect / friendliness",
    ":sun_with_face:": "positive affect / friendliness",
    ":hugging_face:": "positive affect / friendliness",
    ":smiling_face_with_3_hearts:": "positive affect / friendliness",
    ":sunglasses:": "positive affect / friendliness",
    ":call_me_hand:": "positive affect / friendliness",
    ":sunflower:": "positive affect / friendliness",
    ":four_leaf_clover:": "positive affect / friendliness",

    # Humor / softening
    ":sweat_smile:": "humor / softening",
    ":joy:": "humor / softening",
    ":laughing:": "humor / softening",
    ":rofl:": "humor / softening",
    ":see_no_evil:": "humor / softening",
    ":face_with_hand_over_mouth:": "humor / softening",
    ":stuck_out_tongue:": "humor / softening",
    ":stuck_out_tongue_winking_eye:": "humor / softening",
    ":zany_face:": "humor / softening",
    ":disguised_face:": "humor / softening",
    ":nerd_face:": "humor / softening",
    ":clown_face:": "humor / softening",
    ":smirk:": "humor / softening",

    # Affection / support
    ":heart:": "affection / support",
    ":heart_eyes:": "affection / support",
    ":kissing_heart:": "affection / support",
    ":pray:": "affection / support",
    ":hugs:": "affection / support",
    ":tulip:": "affection / support",
    ":hearts:": "affection / support",

    # Information / direction
    ":bulb:": "information / direction",
    ":arrow_right:": "information / direction",
    ":point_right:": "information / direction",
    ":point_up:": "information / direction",
    ":eyes:": "information / direction",
    ":memo:": "information / direction",

    # Concern / uncertainty
    ":thinking_face:": "concern / uncertainty",
    ":confused:": "concern / uncertainty",
    ":worried:": "concern / uncertainty",
    ":grimacing:": "concern / uncertainty",
    ":disappointed:": "concern / uncertainty",
    ":cry:": "concern / uncertainty",
    ":woman_shrugging:": "concern / uncertainty",
    ":woman-shrugging:": "concern / uncertainty",
    ":thinking:": "concern / uncertainty",
    ":frowning_face:": "concern / uncertainty",
    ":scream:": "concern / uncertainty",
    ":cold_sweat:": "concern / uncertainty",
    ":weary:": "concern / uncertainty",
    ":sleepy:": "concern / uncertainty",
    ":hot_face:": "concern / uncertainty",

    # Skepticism / mild negativity
    ":face_with_rolling_eyes:": "skepticism / mild negativity",
    ":roll_eyes:": "skepticism / mild negativity",
    ":unamused:": "skepticism / mild negativity",
    ":upside_down_face:": "skepticism / mild negativity",
    ":no_mouth:": "skepticism / mild negativity",

    # Celebration / seasonal
    ":champagne:": "celebration / seasonal",
    ":christmas_tree:": "celebration / seasonal",
    ":cake:": "celebration / seasonal",
    ":partying_face:": "celebration / seasonal",

    # Objects / symbols / other
    ":coffee:": "objects / symbols / other",
    ":tada:": "objects / symbols / other",
    ":gift:": "objects / symbols / other",
    ":calendar:": "objects / symbols / other",
    ":warning:": "objects / symbols / other",
    ":couch_and_lamp:": "objects / symbols / other",
    ":stew:": "objects / symbols / other",
    ":mask:": "objects / symbols / other",
    ":sandwich:": "objects / symbols / other",
    ":green_salad:": "objects / symbols / other",
    ":hamburger:": "objects / symbols / other",
    ":sleeping:": "objects / symbols / other",
    ":iphone:": "objects / symbols / other",
    ":mountain_bicyclist:": "objects / symbols / other",
    ":mailbox_with_no_mail:": "objects / symbols / other",
    ":apple:": "objects / symbols / other",
}

MANUAL_UNICODE_MAPPING = {
    ":+1:": "👍",
    ":white_check_mark:": "✅",
    ":relaxed:": "☺️",
    ":slightly_smiling_face:": "🙂",
    ":sweat_smile:": "😅",
    ":blush:": "😊",
    ":wink:": "😉",
    ":joy:": "😂",
    ":stuck_out_tongue_winking_eye:": "😜",
    ":heart_eyes:": "😍",
    ":tulip:": "🌷",
    ":tada:": "🎉",
    ":woman_shrugging:": "🤷‍♀️",
    ":woman-shrugging:": "🤷‍♀️",
    ":face_with_rolling_eyes:": "🙄",
    ":roll_eyes:": "🙄",
    ":upside_down_face:": "🙃",
    ":face_with_hand_over_mouth:": "🤭",
    ":no_mouth:": "😶",
    ":nerd_face:": "🤓",
    ":v:": "✌️",
    ":muscle:": "💪",
    ":thinking:": "🤔",
    ":hearts:": "♥️",
    ":bulb:": "💡",
    ":arrow_right:": "➡️",
    ":kissing_heart:": "😘",
    ":rocket:": "🚀",
    ":champagne:": "🍾",
    ":christmas_tree:": "🎄",
    ":cake:": "🎂",
    ":mask:": "😷",
    ":sleeping:": "😴",
    ":sunglasses:": "😎",
    ":call_me_hand:": "🤙",
    ":iphone:": "📱",
    ":sunflower:": "🌻",
    ":partying_face:": "🥳",
    ":clown_face:": "🤡",
    ":cold_sweat:": "😰",
    ":smirk:": "😏",
    ":weary:": "😩",
    ":sleepy:": "😪",
    ":four_leaf_clover:": "🍀",
    ":hot_face:": "🥵",
    ":mountain_bicyclist:": "🚵",
    ":mailbox_with_no_mail:": "📭",
    ":apple:": "🍎",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate public aggregated emoji-shortcode analysis tables."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory. Default: current working directory.",
    )
    parser.add_argument(
        "--direct-path",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv"),
        help="Path to cleaned direct-message CSV, relative to project dir or absolute.",
    )
    parser.add_argument(
        "--group-path",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv"),
        help="Path to cleaned group-message CSV, relative to project dir or absolute.",
    )
    parser.add_argument(
        "--text-column",
        default="text_original",
        help="Text column used for emoji extraction. Default: text_original; falls back to text.",
    )
    parser.add_argument(
        "--out-tables-dir",
        type=Path,
        default=Path("outputs/public/tables/emoji"),
        help="Output directory for public emoji tables.",
    )
    parser.add_argument(
        "--confidential-out-dir",
        type=Path,
        default=Path("outputs/confidential/review_files/emoji"),
        help="Output directory for optional confidential example tables.",
    )
    parser.add_argument(
        "--write-confidential-examples",
        action="store_true",
        help="Write message-level examples with text to outputs/confidential/. Default: off.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV copies of public tables.",
    )
    return parser.parse_args()


def resolve_path(project_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_dir / path


def select_text_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    if "text" in df.columns:
        return "text"
    if "text_clean_lexical" in df.columns:
        return "text_clean_lexical"
    raise ValueError(
        f"No usable text column found. Requested '{requested}'. "
        f"Available columns: {df.columns.tolist()}"
    )


def extract_shortcodes(text: object) -> list[str]:
    if pd.isna(text):
        return []
    candidates = SHORTCODE_PATTERN.findall(str(text))
    return [
        shortcode
        for shortcode in candidates
        if not NON_EMOJI_TIME_PATTERN.match(shortcode)
    ]


def extract_communicative_unicode_emojis(text: object) -> list[str]:
    if emoji_lib is None or pd.isna(text):
        return []
    items = emoji_lib.emoji_list(str(text))
    return [item["emoji"] for item in items if item["emoji"] not in EXCLUDED_UNICODE_SYMBOLS]


def shortcode_to_unicode(shortcode: str) -> str:
    if shortcode in MANUAL_UNICODE_MAPPING:
        return MANUAL_UNICODE_MAPPING[shortcode]
    if emoji_lib is None:
        return ""
    rendered = emoji_lib.emojize(shortcode, language="alias")
    return "" if rendered == shortcode else rendered


def load_text_table(path: Path, direction: str, requested_text_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    text_column = select_text_column(df, requested_text_column)
    out = pd.DataFrame({
        "direction": direction,
        "message_type": MESSAGE_LABELS[direction],
        "id": df["id"] if "id" in df.columns else range(len(df)),
        "conversation_id": df["conversation_id"] if "conversation_id" in df.columns else "",
        "text": df[text_column].fillna("").astype(str),
    })
    return out


def analyze_direction(df: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_utterances = len(df)

    all_shortcodes: list[str] = []
    shortcode_example_rows = []
    unicode_emojis: list[str] = []
    utterances_with_shortcodes = 0
    utterances_with_unicode = 0

    for _, row in df.iterrows():
        text = row["text"]
        shortcodes = extract_shortcodes(text)
        unicodes = extract_communicative_unicode_emojis(text)

        if shortcodes:
            utterances_with_shortcodes += 1
            all_shortcodes.extend(shortcodes)
            shortcode_example_rows.append({
                "direction": direction,
                "message_type": MESSAGE_LABELS[direction],
                "id": row.get("id", ""),
                "conversation_id": row.get("conversation_id", ""),
                "emoji_shortcodes": "; ".join(shortcodes),
                "emoji_shortcode_count": len(shortcodes),
                "text": text,
            })

        if unicodes:
            utterances_with_unicode += 1
            unicode_emojis.extend(unicodes)

    shortcode_counts = Counter(all_shortcodes)
    unicode_counts = Counter(unicode_emojis)

    shortcode_freq = pd.DataFrame(
        shortcode_counts.most_common(),
        columns=["emoji_shortcode", "count"],
    )
    if shortcode_freq.empty:
        shortcode_freq = pd.DataFrame(columns=["emoji_shortcode", "count", "percentage"])
    else:
        total_shortcodes = int(shortcode_freq["count"].sum())
        shortcode_freq["percentage"] = (shortcode_freq["count"] / total_shortcodes * 100).round(2)

    shortcode_freq.insert(0, "direction", direction)
    shortcode_freq.insert(1, "message_type", MESSAGE_LABELS[direction])

    unicode_freq = pd.DataFrame(
        unicode_counts.most_common(),
        columns=["unicode_emoji", "count"],
    )
    if unicode_freq.empty:
        unicode_freq = pd.DataFrame(columns=["unicode_emoji", "count", "percentage"])
    else:
        total_unicode = int(unicode_freq["count"].sum())
        unicode_freq["percentage"] = (unicode_freq["count"] / total_unicode * 100).round(2)
    unicode_freq.insert(0, "direction", direction)
    unicode_freq.insert(1, "message_type", MESSAGE_LABELS[direction])

    total_shortcodes = int(shortcode_freq["count"].sum()) if "count" in shortcode_freq.columns else 0
    total_unicode = int(unicode_freq["count"].sum()) if "count" in unicode_freq.columns else 0

    summary = pd.DataFrame([
        {
            "direction": direction,
            "message_type": MESSAGE_LABELS[direction],
            "total_utterances": total_utterances,
            "utterances_with_communicative_unicode_emojis": utterances_with_unicode,
            "total_communicative_unicode_emojis": total_unicode,
            "unique_communicative_unicode_emojis": len(unicode_counts),
            "utterances_with_emoji_shortcodes": utterances_with_shortcodes,
            "total_emoji_shortcodes_before_artifact_exclusion": total_shortcodes,
            "unique_emoji_shortcodes_before_artifact_exclusion": len(shortcode_counts),
            "percentage_utterances_with_emoji_shortcodes": round((utterances_with_shortcodes / total_utterances * 100), 2) if total_utterances else 0,
            "notes": "Aggregated counts only; no message text included in public outputs.",
        }
    ])

    examples = pd.DataFrame(shortcode_example_rows)
    return summary, shortcode_freq, unicode_freq, examples


def build_codebook(shortcode_freq: pd.DataFrame) -> pd.DataFrame:
    codebook = shortcode_freq.copy()
    if codebook.empty:
        return pd.DataFrame(columns=[
            "direction", "message_type", "emoji_shortcode", "count", "percentage",
            "category", "unicode_display", "is_excluded_artifact"
        ])

    codebook["is_excluded_artifact"] = (
        codebook["emoji_shortcode"].isin(NON_EMOJI_SHORTCODE_VALUES)
        | codebook["emoji_shortcode"].str.match(NUMERIC_SHORTCODE_PATTERN)
    )
    codebook["category"] = codebook["emoji_shortcode"].map(EMOJI_CATEGORY_MAPPING).fillna("uncategorized")
    codebook.loc[codebook["is_excluded_artifact"], "category"] = "excluded non-emoji artefact"
    codebook["unicode_display"] = codebook["emoji_shortcode"].apply(shortcode_to_unicode)
    return codebook


def build_category_summary(codebook: pd.DataFrame) -> pd.DataFrame:
    retained = codebook[~codebook["is_excluded_artifact"]].copy()
    if retained.empty:
        return pd.DataFrame(columns=[
            "direction", "message_type", "category", "total_count", "unique_emoji_shortcodes",
            "percentage", "representative_shortcodes", "visual_examples"
        ])

    retained["shortcode_with_count"] = retained["emoji_shortcode"] + " (" + retained["count"].astype(int).astype(str) + ")"
    retained["unicode_with_count"] = retained.apply(
        lambda row: f"{row['unicode_display']} ({int(row['count'])})" if row["unicode_display"] else "",
        axis=1,
    )

    top_examples = (
        retained.sort_values(["direction", "category", "count"], ascending=[True, True, False])
        .groupby(["direction", "category"])
        .head(3)
    )
    examples = (
        top_examples.groupby(["direction", "category"], as_index=False)
        .agg(
            representative_shortcodes=("shortcode_with_count", lambda x: "; ".join(x)),
            visual_examples=("unicode_with_count", lambda x: "; ".join([item for item in x if item])),
        )
    )

    summary = (
        retained.groupby(["direction", "message_type", "category"], as_index=False)
        .agg(
            total_count=("count", "sum"),
            unique_emoji_shortcodes=("emoji_shortcode", "nunique"),
        )
    )
    totals = summary.groupby("direction")["total_count"].transform("sum")
    summary["percentage"] = (summary["total_count"] / totals * 100).round(2)
    summary = summary.merge(examples, on=["direction", "category"], how="left")
    summary = summary.sort_values(["direction", "total_count"], ascending=[True, False]).reset_index(drop=True)
    return summary


def build_direct_group_category_comparison(category_summary: pd.DataFrame) -> pd.DataFrame:
    direct = category_summary[category_summary["direction"] == "direct"].copy()
    group = category_summary[category_summary["direction"] == "group"].copy()

    direct = direct.rename(columns={
        "total_count": "direct_total_count",
        "unique_emoji_shortcodes": "direct_unique_emoji_shortcodes",
        "percentage": "direct_percentage",
        "representative_shortcodes": "direct_representative_shortcodes",
        "visual_examples": "direct_visual_examples",
    })[[
        "category", "direct_total_count", "direct_unique_emoji_shortcodes",
        "direct_percentage", "direct_representative_shortcodes", "direct_visual_examples"
    ]]

    group = group.rename(columns={
        "total_count": "group_total_count",
        "unique_emoji_shortcodes": "group_unique_emoji_shortcodes",
        "percentage": "group_percentage",
        "representative_shortcodes": "group_representative_shortcodes",
        "visual_examples": "group_visual_examples",
    })[[
        "category", "group_total_count", "group_unique_emoji_shortcodes",
        "group_percentage", "group_representative_shortcodes", "group_visual_examples"
    ]]

    comparison = direct.merge(group, on="category", how="outer")
    for col in [
        "direct_total_count", "direct_unique_emoji_shortcodes", "direct_percentage",
        "group_total_count", "group_unique_emoji_shortcodes", "group_percentage",
    ]:
        comparison[col] = pd.to_numeric(comparison[col], errors="coerce").fillna(0)

    for col in [
        "direct_representative_shortcodes", "direct_visual_examples",
        "group_representative_shortcodes", "group_visual_examples",
    ]:
        comparison[col] = comparison[col].fillna("")

    comparison["percentage_point_difference_group_minus_direct"] = (
        comparison["group_percentage"] - comparison["direct_percentage"]
    ).round(2)
    comparison["combined_total_count"] = comparison["direct_total_count"] + comparison["group_total_count"]
    return comparison.sort_values("combined_total_count", ascending=False).reset_index(drop=True)


def build_publication_ready_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    pub = comparison.copy()
    pub["Direct messages, n (%)"] = (
        pub["direct_total_count"].astype(int).astype(str)
        + " (" + pub["direct_percentage"].round(2).astype(str) + "%)"
    )
    pub["Group messages, n (%)"] = (
        pub["group_total_count"].astype(int).astype(str)
        + " (" + pub["group_percentage"].round(2).astype(str) + "%)"
    )
    pub = pub.rename(columns={
        "category": "Emoji shortcode category",
        "percentage_point_difference_group_minus_direct": "Difference, percentage points",
    })
    return pub[[
        "Emoji shortcode category",
        "Direct messages, n (%)",
        "Group messages, n (%)",
        "Difference, percentage points",
        "direct_representative_shortcodes",
        "group_representative_shortcodes",
    ]]


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    direct_path = resolve_path(project_dir, args.direct_path)
    group_path = resolve_path(project_dir, args.group_path)
    out_tables_dir = resolve_path(project_dir, args.out_tables_dir)
    confidential_out_dir = resolve_path(project_dir, args.confidential_out_dir)

    out_tables_dir.mkdir(parents=True, exist_ok=True)

    direct_df = load_text_table(direct_path, "direct", args.text_column)
    group_df = load_text_table(group_path, "group", args.text_column)

    outputs = [analyze_direction(direct_df, "direct"), analyze_direction(group_df, "group")]

    summary = pd.concat([item[0] for item in outputs], ignore_index=True)
    shortcode_frequency = pd.concat([item[1] for item in outputs], ignore_index=True)
    unicode_frequency = pd.concat([item[2] for item in outputs], ignore_index=True)
    examples = pd.concat([item[3] for item in outputs], ignore_index=True)

    codebook = build_codebook(shortcode_frequency)
    category_summary = build_category_summary(codebook)
    category_comparison = build_direct_group_category_comparison(category_summary)
    publication_ready = build_publication_ready_comparison(category_comparison)

    # Update retained shortcode counts after artefact exclusion in the summary.
    retained_counts = (
        codebook[~codebook["is_excluded_artifact"]]
        .groupby("direction")
        .agg(
            total_retained_emoji_shortcodes=("count", "sum"),
            unique_retained_emoji_shortcodes=("emoji_shortcode", "nunique"),
        )
        .reset_index()
    )
    summary = summary.merge(retained_counts, on="direction", how="left")
    summary[["total_retained_emoji_shortcodes", "unique_retained_emoji_shortcodes"]] = summary[[
        "total_retained_emoji_shortcodes", "unique_retained_emoji_shortcodes"
    ]].fillna(0).astype(int)

    workbook_path = out_tables_dir / "emoji_analysis_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        shortcode_frequency.to_excel(writer, sheet_name="shortcode_frequency_raw", index=False)
        unicode_frequency.to_excel(writer, sheet_name="unicode_frequency", index=False)
        codebook.to_excel(writer, sheet_name="shortcode_codebook", index=False)
        category_summary.to_excel(writer, sheet_name="category_summary", index=False)
        category_comparison.to_excel(writer, sheet_name="category_comparison", index=False)
        publication_ready.to_excel(writer, sheet_name="publication_ready", index=False)

    if args.write_csv:
        summary.to_csv(out_tables_dir / "emoji_summary_by_corpus.csv", index=False, encoding="utf-8")
        shortcode_frequency.to_csv(out_tables_dir / "emoji_shortcode_frequency_by_corpus.csv", index=False, encoding="utf-8")
        codebook.to_csv(out_tables_dir / "emoji_shortcode_codebook.csv", index=False, encoding="utf-8")
        category_summary.to_csv(out_tables_dir / "emoji_category_summary_by_corpus.csv", index=False, encoding="utf-8")
        category_comparison.to_csv(out_tables_dir / "emoji_category_direct_group_comparison.csv", index=False, encoding="utf-8")
        publication_ready.to_csv(out_tables_dir / "emoji_category_direct_group_comparison_publication_ready.csv", index=False, encoding="utf-8")

    if args.write_confidential_examples:
        confidential_out_dir.mkdir(parents=True, exist_ok=True)
        examples.to_excel(confidential_out_dir / "emoji_shortcode_examples_with_text.xlsx", index=False)

    print("Saved public emoji analysis workbook:")
    print(workbook_path)
    print("Summary:")
    print(summary.to_string(index=False))
    if args.write_confidential_examples:
        print("Saved confidential examples to:")
        print(confidential_out_dir)


if __name__ == "__main__":
    main()
