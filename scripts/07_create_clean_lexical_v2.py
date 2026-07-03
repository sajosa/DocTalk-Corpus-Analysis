#!/usr/bin/env python3
"""
Create lexical v2 tables for collocation analysis.

v2 changes:
1. Normalize therapy group multiword expressions:
   MT Gruppe -> MT_Gruppe
   KT Gruppe -> KT_Gruppe
   GT Gruppe -> GT_Gruppe

2. Normalize explicit no-task status expressions:
   kein aktives Todo -> kein_Todo
   kein aktuelles Todo -> kein_Todo
   kein_aktives_Todo -> kein_Todo

3. Normalize selected sentence-initial pronouns/function words:
   Ich -> ich
   Du -> du
   Mir -> mir
   etc.

4. Create a content_text_v2 column with selected high-frequency German
   function words removed for content-oriented collocation analysis.

Inputs:
    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Outputs:
    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical_v2.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical_v2.csv
    outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical_v2.csv
    outputs/public/tables/collocations_v2/marker_presence_check_clean_lexical_v2.xlsx

Usage from project root:
    python scripts/07_create_clean_lexical_v2.py
"""

from pathlib import Path
import re
import pandas as pd


PROJECT_DIR = Path.cwd()

DIRECT_IN = PROJECT_DIR / "outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv"
GROUP_IN = PROJECT_DIR / "outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv"

DIRECT_OUT = PROJECT_DIR / "outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical_v2.csv"
GROUP_OUT = PROJECT_DIR / "outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical_v2.csv"

COMBINED_OUT = PROJECT_DIR / "outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical_v2.csv"
MARKER_OUT = PROJECT_DIR / "outputs/public/tables/collocations_v2/marker_presence_check_clean_lexical_v2.xlsx"

TEXT_COL = "text_clean_lexical"


# -----------------------------------------------------------------------------
# Rule-based v2 normalization
# -----------------------------------------------------------------------------

THERAPY_GROUP_RULES = [
    (r"\bMT\s+Gruppe\b", "MT_Gruppe"),
    (r"\bKT\s+Gruppe\b", "KT_Gruppe"),
    (r"\bGT\s+Gruppe\b", "GT_Gruppe"),
]

TODO_STATUS_RULES = [
    # Hashtag artefact variants, e.g. original "#kein to do"
    (r"\bHashtag_kein\s+Todo\b", "kein_Todo"),
    (r"\bHashtag_keine\s+Todo\b", "kein_Todo"),

    # Underscore variants
    (r"\bkein_aktives_Todo\b", "kein_Todo"),
    (r"\bkein_aktives_ToDo\b", "kein_Todo"),
    (r"\bkein_aktuelles_Todo\b", "kein_Todo"),
    (r"\bkein_aktuelles_ToDo\b", "kein_Todo"),

    # Whitespace variants
    (r"\bkein\s+aktives\s+Todo\b", "kein_Todo"),
    (r"\bkein\s+aktives\s+ToDo\b", "kein_Todo"),
    (r"\bkein\s+aktuelles\s+Todo\b", "kein_Todo"),
    (r"\bkein\s+aktuelles\s+ToDo\b", "kein_Todo"),
]

CASE_NORMALIZATION_RULES = [
    # Selected sentence-initial pronouns/function words.
    # Do not lowercase the full text because protected markers must remain stable.
    (r"\bIch\b", "ich"),
    (r"\bDu\b", "du"),
    (r"\bMir\b", "mir"),
    (r"\bMich\b", "mich"),
    (r"\bDir\b", "dir"),
    (r"\bDich\b", "dich"),
    (r"\bWir\b", "wir"),
    (r"\bUns\b", "uns"),
    (r"\bSich\b", "sich"),
]


# Conservative German/function-word stoplist for content-oriented collocation.
# Interactionally relevant words such as ich/du/mir are removed from content_text_v2,
# but preserved in interaction_text_v2.
CONTENT_STOPWORDS = {
    "der", "die", "das",
    "den", "dem", "des",
    "ein", "eine", "einer", "einem", "einen", "eines",
    "und", "oder", "aber",
    "in", "im", "am", "an", "auf", "aus", "bei", "mit", "nach", "von", "vor", "zu", "zur", "zum",
    "für", "über", "unter", "zwischen",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten",
    "sein", "gewesen",
    "es", "er", "sie", "wir", "ihr",
    "ich", "du", "mir", "mich", "dir", "dich", "uns", "sich",
    "da", "dann", "noch", "auch", "schon", "nur", "so", "mal",
    "wie", "was", "wer", "wo", "wann",
    "dass", "wenn", "weil", "als",
}


MARKERS_TO_CHECK = [
    "PatName",
    "Hashtag_PatName",
    "KolName",
    "Mention_KolName",
    "Übergabe",
    "WE",
    "Todo",
    "kein_Todo",
    "kein_aktives_Todo",
    "kein",
    "Rückmeldung",
    "Gruppe",
    "MT",
    "GT",
    "KT",
    "MT_Gruppe",
    "GT_Gruppe",
    "KT_Gruppe",
    "Raum",
    "ÖGD",
]


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_therapy_groups(text: str) -> str:
    """
    Normalize recurrent therapy group multiword expressions.
    """
    text = safe_text(text)
    for pattern, replacement in THERAPY_GROUP_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_todo_status(text: str) -> str:
    """
    Normalize explicit no-task status expressions.

    These variants all express absence of an active/pending task and are
    therefore mapped to the established marker kein_Todo.
    """
    text = safe_text(text)
    for pattern, replacement in TODO_STATUS_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_case_selected(text: str) -> str:
    """
    Normalize selected sentence-initial function words/pronouns without lowercasing
    protected markers such as PatName, KolName, WE, ÖGD, MT_Gruppe.
    """
    text = safe_text(text)
    for pattern, replacement in CASE_NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def remove_content_stopwords(text: str) -> str:
    """
    Token-based removal.
    Keeps protected markers with underscores and clinical abbreviations.
    """
    text = safe_text(text)
    tokens = text.split()
    kept = []

    for tok in tokens:
        # Strip punctuation only for stopword comparison, but keep original token if retained.
        comparable = tok.strip(".,;:!?()[]{}\"'„“”").lower()

        if comparable in CONTENT_STOPWORDS:
            continue

        kept.append(tok)

    return " ".join(kept)


def add_v2_columns(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    df = df.copy()
    df["direction"] = direction

    if TEXT_COL not in df.columns:
        raise ValueError(f"Required column not found: {TEXT_COL}. Available columns: {df.columns.tolist()}")

    df["text_clean_lexical_v2"] = (
        df[TEXT_COL]
        .apply(normalize_therapy_groups)
        .apply(normalize_todo_status)
        .apply(normalize_case_selected)
    )

    # Content view: MWE normalization + no-task normalization + selected case normalization
    # + stopword removal.
    df["content_text_v2"] = df["text_clean_lexical_v2"].apply(remove_content_stopwords)

    # Interaction view for now: v2 normalization only, without content stopword removal.
    # This preserves interactionally relevant words such as ich, du, bitte, danke, hallo, liebe.
    df["interaction_text_v2"] = df["text_clean_lexical_v2"]

    return df


def count_marker(text: str, marker: str) -> int:
    text = safe_text(text)

    # Exact token-ish match; underscores and umlauts are preserved.
    pattern = rf"(?<!\w){re.escape(marker)}(?!\w)"
    return len(re.findall(pattern, text))


def marker_presence_check(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    rows = []

    for marker in MARKERS_TO_CHECK:
        counts = df[text_col].apply(lambda x: count_marker(x, marker))
        rows.append(
            {
                "marker": marker,
                "messages_with_marker": int((counts > 0).sum()),
                "total_occurrences": int(counts.sum()),
            }
        )

    return pd.DataFrame(rows)


def main():
    if not DIRECT_IN.exists():
        raise FileNotFoundError(f"Direct input not found: {DIRECT_IN}")
    if not GROUP_IN.exists():
        raise FileNotFoundError(f"Group input not found: {GROUP_IN}")

    d = pd.read_csv(DIRECT_IN)
    g = pd.read_csv(GROUP_IN)

    print("Loaded Direct:", d.shape)
    print("Loaded Group:", g.shape)

    d_v2 = add_v2_columns(d, "direct")
    g_v2 = add_v2_columns(g, "group")

    DIRECT_OUT.parent.mkdir(parents=True, exist_ok=True)
    GROUP_OUT.parent.mkdir(parents=True, exist_ok=True)

    d_v2.to_csv(DIRECT_OUT, index=False)
    g_v2.to_csv(GROUP_OUT, index=False)

    combined = pd.concat([d_v2, g_v2], ignore_index=True, sort=False)
    combined.to_csv(COMBINED_OUT, index=False)

    marker_original = marker_presence_check(combined, "text_clean_lexical")
    marker_v2 = marker_presence_check(combined, "text_clean_lexical_v2")
    marker_content_v2 = marker_presence_check(combined, "content_text_v2")

    MARKER_OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(MARKER_OUT, engine="openpyxl") as writer:
        marker_original.to_excel(writer, sheet_name="original_clean_lexical", index=False)
        marker_v2.to_excel(writer, sheet_name="text_clean_lexical_v2", index=False)
        marker_content_v2.to_excel(writer, sheet_name="content_text_v2", index=False)

    print("Saved:")
    print(" -", DIRECT_OUT)
    print(" -", GROUP_OUT)
    print(" -", COMBINED_OUT)
    print(" -", MARKER_OUT)

    print("\nMarker check v2:")
    print(marker_v2)


if __name__ == "__main__":
    main()