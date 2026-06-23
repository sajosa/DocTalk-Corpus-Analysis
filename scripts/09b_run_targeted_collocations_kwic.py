"""
Targeted collocation and KWIC analysis for clinical chat corpora.

Purpose
-------
Use this script after keyness analysis to examine the local lexical neighbourhoods
of selected core-result/keyness items. It is designed for an utterance-level table
with Direct and Group messages and optional Content/Interaction views.

Expected input
--------------
A pandas DataFrame or CSV/XLSX with at least:
- one row per utterance/message
- a modality column, e.g. direction = "direct" or "group"
- one or two cleaned text columns, e.g. content_text and interaction_text
- optional metadata columns, e.g. conversation_id, utterance_id, speaker, timestamp

Recommended defaults
--------------------
window_size = 5 tokens left/right
min_freq = 3
association metric = dice and window-based log-likelihood (G2)
PMI is computed only as an optional supplementary metric and should not be used alone.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import sys

import pandas as pd


# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------

DEFAULT_ANCHORS = [
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

# Map known anchors/collocates to cautious initial categories.
# This is intentionally conservative; final interpretation should be KWIC-validated.
CATEGORY_HINTS = {
    "PatName": "Patient reference",
    "Hashtag_PatName": "Patient reference",
    "KolName": "Colleague reference/addressivity",
    "Mention_KolName": "Colleague reference/addressivity",
    "Mention_All": "Colleague reference/addressivity",
    "Übergabe": "Handover/status",
    "WE": "Handover/status; Time/scheduling",
    "Todo": "Handover/status",
    "ToDo": "Handover/status",
    "kein_Todo": "Handover/status",
    "kein": "Handover/status; Clinical/content negation marker - check KWIC",
    "Rückmeldung": "Documentation/communication logistics; Therapy/group context - check KWIC",
    "Gruppe": "Therapy/group context",
    "MT": "Therapy/group context",
    "GT": "Therapy/group context",
    "KT": "Therapy/group context",
    "anwesend": "Therapy/group context",
    "Raum": "Documentation/communication logistics; Time/scheduling - check KWIC",
    "ÖGD": "Clinical content",
    "Brief": "Documentation/communication logistics",
    "Datum": "Time/scheduling",
    "morgen": "Time/scheduling",
    "heute": "Time/scheduling",
    "Uhr": "Time/scheduling",
    "uhr": "Time/scheduling",
    "fertig": "Handover/status; Documentation/communication logistics - check KWIC",
    "QuestionMark": "Interaction/politeness",
    "bitte": "Interaction/politeness",
    "danke": "Interaction/politeness",
    "gerne": "Interaction/politeness",
    "Hallo": "Interaction/politeness",
    "Hi": "Interaction/politeness",
    "Moin": "Interaction/politeness",
    "Guten": "Interaction/politeness",
    "Morgen": "Interaction/politeness; Time/scheduling - check KWIC",
    "Atorvastatin": "Clinical content",
    "Suizidalität": "Clinical content",
    "Kolo": "Clinical content",
    "BE": "Clinical content",
    "Kalinor": "Clinical content",
}

ARTICLE_PROFILE_SEEDS = [
    {
        "Collocation profile": "Patient anchoring around case references",
        "Expected anchors": "PatName; Hashtag_PatName",
        "Dominant modality": "Both / mixed",
        "Communicative interpretation": "Patient markers structure the local discourse and should be interpreted as case anchoring rather than as clinical content on their own.",
    },
    {
        "Collocation profile": "Handover and task-status bundles",
        "Expected anchors": "Übergabe; WE; Todo; kein_Todo; kein",
        "Dominant modality": "Group",
        "Communicative interpretation": "Collocates around handover and To-do markers can show whether group messages function as shared continuity-of-care/status spaces.",
    },
    {
        "Collocation profile": "Therapy/group documentation and attendance",
        "Expected anchors": "Rückmeldung; Gruppe; MT; GT; KT; anwesend",
        "Dominant modality": "Group",
        "Communicative interpretation": "Collocates should be checked for attendance, therapy group feedback, and collective documentation routines.",
    },
    {
        "Collocation profile": "Temporal coordination and scheduling",
        "Expected anchors": "morgen; heute; Datum; Uhr; WE",
        "Dominant modality": "Direct",
        "Communicative interpretation": "Temporal collocates can support the interpretation of direct messages as bilateral scheduling and rapid coordination.",
    },
    {
        "Collocation profile": "Clinical/procedural clarification",
        "Expected anchors": "ÖGD; Kolo; BE; Atorvastatin; Suizidalität",
        "Dominant modality": "Direct / mixed",
        "Communicative interpretation": "Clinical collocates should only be interpreted substantively after KWIC review because patient anchors and procedures may bundle several functions.",
    },
    {
        "Collocation profile": "Interactional work and addressivity",
        "Expected anchors": "KolName; Mention_KolName; Hallo; Hi; danke; bitte",
        "Dominant modality": "Direct",
        "Communicative interpretation": "Addressivity and politeness collocates can show interpersonal alignment, request mitigation, and bilateral coordination.",
    },
    {
        "Collocation profile": "Documentation, communication, and resource logistics",
        "Expected anchors": "Raum; Brief; Rückmeldung; Link; WIKI",
        "Dominant modality": "Direct / Group depending on item",
        "Communicative interpretation": "Resource and documentation collocates should be interpreted as logistics unless KWIC supports another primary function.",
    },
]


# -----------------------------------------------------------------------------
# 2) Tokenization and utility functions
# -----------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9_#/@:+\-]+|[!?]", flags=re.UNICODE)


def tokenize(text: object, lowercase: bool = False, mode: str = "regex") -> list[str]:
    """Tokenize cleaned chat text while preserving placeholders such as PatName.

    Parameters
    ----------
    text:
        Input text.
    lowercase:
        If True, lowercase all tokens. Keep False for placeholder-sensitive analysis.
    mode:
        "regex" for robust token extraction, "whitespace" if your text is already tokenized.
    """
    if pd.isna(text):
        return []
    text = str(text)
    if mode == "whitespace":
        tokens = [t.strip() for t in text.split() if t.strip()]
    else:
        tokens = TOKEN_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def normalize_direction(value: object) -> str:
    """Normalize modality labels to direct/group."""
    v = str(value).strip().lower()
    if v in {"direct", "dm", "direct_message", "direct messages", "direkt", "direktnachrichten"}:
        return "direct"
    if v in {"group", "gm", "group_message", "group messages", "gruppe", "gruppennachrichten"}:
        return "group"
    return v


def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


def g2_log_likelihood(k11: int, k12: int, k21: int, k22: int) -> float:
    """Dunning-style G2 log-likelihood for a 2x2 table.

    Table layout:
      k11 = collocate occurrences inside anchor windows
      k12 = non-collocate tokens inside anchor windows
      k21 = collocate occurrences outside anchor windows
      k22 = non-collocate tokens outside anchor windows
    """
    total = k11 + k12 + k21 + k22
    if total == 0:
        return 0.0
    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    expected = [
        row1 * col1 / total,
        row1 * col2 / total,
        row2 * col1 / total,
        row2 * col2 / total,
    ]
    observed = [k11, k12, k21, k22]
    score = 0.0
    for obs, exp in zip(observed, expected):
        if obs > 0 and exp > 0:
            score += obs * math.log(obs / exp)
    return 2 * score


def infer_category(anchor: str, collocate: str) -> str:
    """Return a cautious category hint based on anchor/collocate dictionaries."""
    a = CATEGORY_HINTS.get(anchor)
    c = CATEGORY_HINTS.get(collocate)
    if a and c and a != c:
        return f"primary: {a}; secondary/collocate: {c}"
    return a or c or "To review"


def make_interpretation_note(anchor: str, collocate: str, direction: str, view: str) -> str:
    """Generate a cautious note for review tables; never replaces KWIC review."""
    category = infer_category(anchor, collocate)
    notes = []
    if "check KWIC" in category or category == "To review":
        notes.append("requires KWIC review before interpretation")
    if anchor == "kein":
        notes.append("negation marker validated as genuine; check whether it negates To-do, status, or clinical content")
    if anchor == "WE" or collocate == "WE":
        notes.append("WE = weekend; in handover contexts code primarily as Handover/status, secondarily Time/scheduling")
    if anchor == "ÖGD" or collocate == "ÖGD":
        notes.append("ÖGD = Ösophagusgastroskopie; interpret as clinical/procedural content")
    if anchor == "Raum" or collocate == "Raum":
        notes.append("Raum = room/resource coordination, not therapy group context")
    if not notes:
        notes.append(f"candidate pattern in {direction}/{view}; verify representative KWIC examples")
    return "; ".join(notes)


# -----------------------------------------------------------------------------
# 3) Optional anchor extraction from the condensed JMIR keyness table
# -----------------------------------------------------------------------------


def load_core_items_as_anchors(
    keyness_xlsx: str | Path,
    sheet_name: str = "Representative_items",
    role_col: str = "Role",
    item_col: str = "Item",
    keep_multiword_items: bool = False,
) -> list[str]:
    """Extract anchors from the condensed keyness result table.

    By default, multiword items are split into component tokens because the collocation
    algorithm is token-anchor based. Set keep_multiword_items=True only if you use the
    KWIC phrase search functions for n-grams.
    """
    path = Path(keyness_xlsx)
    if not path.exists():
        raise FileNotFoundError(f"Keyness table not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    if role_col in df.columns:
        df = df[df[role_col].astype(str).str.lower().eq("core result")]

    anchors: set[str] = set()
    for item in df[item_col].dropna().astype(str):
        if keep_multiword_items:
            anchors.add(item)
        else:
            for token in item.split():
                anchors.add(token)
    return sorted(anchors)


# -----------------------------------------------------------------------------
# 4) Main targeted collocation analysis
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CollocationConfig:
    direction_col: str = "direction"
    text_cols: dict[str, str] | None = None
    metadata_cols: tuple[str, ...] = ("conversation_id", "utterance_id", "speaker", "timestamp")
    anchors: tuple[str, ...] = tuple(DEFAULT_ANCHORS)
    window_size: int = 5
    min_freq: int = 3
    lowercase: bool = False
    tokenization_mode: str = "regex"
    include_anchor_as_collocate: bool = False

    def resolved_text_cols(self) -> dict[str, str]:
        if self.text_cols is not None:
            return self.text_cols
        return {
            "content": "content_text",
            "interaction": "interaction_text",
        }


def targeted_collocations(
    utterances_df: pd.DataFrame,
    config: CollocationConfig,
) -> pd.DataFrame:
    """Compute targeted collocations by direction and view.

    The algorithm does not cross utterance boundaries. For each anchor occurrence,
    it counts collocate tokens within ±window_size. It also computes Dice, PMI, and
    a window-based G2 log-likelihood.
    """
    df = utterances_df.copy()
    df[config.direction_col] = df[config.direction_col].map(normalize_direction)

    anchors = set(config.anchors)
    if config.lowercase:
        anchors = {a.lower() for a in anchors}

    out_rows = []

    for view, text_col in config.resolved_text_cols().items():
        if text_col not in df.columns:
            print(f"Skipping view '{view}': missing text column '{text_col}'")
            continue

        for direction in ["direct", "group"]:
            sub = df[df[config.direction_col].eq(direction)].copy()
            if sub.empty:
                continue

            tokenized_docs = [
                tokenize(text, lowercase=config.lowercase, mode=config.tokenization_mode)
                for text in sub[text_col]
            ]
            corpus_token_counts = Counter(t for doc in tokenized_docs for t in doc)
            corpus_n_tokens = sum(corpus_token_counts.values())
            if corpus_n_tokens == 0:
                continue

            # Per-anchor counters
            anchor_counts = Counter()
            cooc_counts: dict[str, Counter] = defaultdict(Counter)
            window_token_counts = Counter()

            for tokens in tokenized_docs:
                n = len(tokens)
                if n == 0:
                    continue
                for i, tok in enumerate(tokens):
                    if tok not in anchors:
                        continue
                    anchor = tok
                    anchor_counts[anchor] += 1
                    left = max(0, i - config.window_size)
                    right = min(n, i + config.window_size + 1)
                    window_tokens = tokens[left:i] + tokens[i + 1 : right]
                    if not config.include_anchor_as_collocate:
                        window_tokens = [w for w in window_tokens if w != anchor]
                    window_token_counts[anchor] += len(window_tokens)
                    cooc_counts[anchor].update(window_tokens)

            for anchor, coll_counter in cooc_counts.items():
                anchor_n = anchor_counts[anchor]
                anchor_window_n = window_token_counts[anchor]
                for collocate, freq in coll_counter.items():
                    if freq < config.min_freq:
                        continue
                    collocate_n = corpus_token_counts[collocate]
                    dice = (2 * freq / (anchor_n + collocate_n)) if (anchor_n + collocate_n) else 0.0
                    # PMI is supplementary only; use with caution for low-frequency items.
                    pmi = math.log2((freq * corpus_n_tokens) / (anchor_n * collocate_n)) if anchor_n and collocate_n else 0.0

                    k11 = freq
                    k12 = max(anchor_window_n - freq, 0)
                    k21 = max(collocate_n - freq, 0)
                    k22 = max(corpus_n_tokens - anchor_window_n - k21, 0)
                    ll_g2 = g2_log_likelihood(k11, k12, k21, k22)

                    out_rows.append(
                        {
                            "anchor": anchor,
                            "collocate": collocate,
                            "direction": direction,
                            "view": view,
                            "window_size": config.window_size,
                            "frequency": int(freq),
                            "anchor_frequency": int(anchor_n),
                            "collocate_frequency": int(collocate_n),
                            "association_metric": "dice",
                            "association_score": dice,
                            "ll_g2": ll_g2,
                            "pmi_supplementary": pmi,
                            "possible_category": infer_category(anchor, collocate),
                            "interpretation_note": make_interpretation_note(anchor, collocate, direction, view),
                        }
                    )

    result = pd.DataFrame(out_rows)
    if result.empty:
        return result

    # Sort for interpretation: within each anchor/direction/view, frequent and strongly associated patterns first.
    result = result.sort_values(
        by=["anchor", "direction", "view", "frequency", "ll_g2", "association_score"],
        ascending=[True, True, True, False, False, False],
    ).reset_index(drop=True)
    return result


# -----------------------------------------------------------------------------
# 5) KWIC / Concordance functions
# -----------------------------------------------------------------------------


def kwic_for_anchor(
    utterances_df: pd.DataFrame,
    anchor: str,
    text_col: str,
    direction_col: str = "direction",
    direction: str | None = None,
    window_size: int = 8,
    lowercase: bool = False,
    tokenization_mode: str = "regex",
    metadata_cols: Sequence[str] = ("conversation_id", "utterance_id", "speaker", "timestamp"),
    max_examples: int | None = 100,
) -> pd.DataFrame:
    """Return KWIC rows for a single token anchor."""
    df = utterances_df.copy()
    df[direction_col] = df[direction_col].map(normalize_direction)
    if direction is not None:
        df = df[df[direction_col].eq(normalize_direction(direction))].copy()

    search_anchor = anchor.lower() if lowercase else anchor
    rows = []
    for _, row in df.iterrows():
        tokens = tokenize(row.get(text_col), lowercase=lowercase, mode=tokenization_mode)
        for i, tok in enumerate(tokens):
            if tok != search_anchor:
                continue
            left_tokens = tokens[max(0, i - window_size) : i]
            right_tokens = tokens[i + 1 : min(len(tokens), i + window_size + 1)]
            rec = {
                "anchor": anchor,
                "direction": row.get(direction_col),
                "left_context": " ".join(left_tokens),
                "node": tok,
                "right_context": " ".join(right_tokens),
                "full_text": row.get(text_col),
            }
            for col in metadata_cols:
                if col in row.index:
                    rec[col] = row.get(col)
            rows.append(rec)
            if max_examples is not None and len(rows) >= max_examples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def kwic_for_anchor_collocate(
    utterances_df: pd.DataFrame,
    anchor: str,
    collocate: str,
    text_col: str,
    direction_col: str = "direction",
    direction: str | None = None,
    collocation_window: int = 5,
    kwic_window: int = 10,
    lowercase: bool = False,
    tokenization_mode: str = "regex",
    metadata_cols: Sequence[str] = ("conversation_id", "utterance_id", "speaker", "timestamp"),
    max_examples: int | None = 50,
) -> pd.DataFrame:
    """Return KWIC rows where collocate occurs within ±collocation_window of anchor."""
    df = utterances_df.copy()
    df[direction_col] = df[direction_col].map(normalize_direction)
    if direction is not None:
        df = df[df[direction_col].eq(normalize_direction(direction))].copy()

    search_anchor = anchor.lower() if lowercase else anchor
    search_collocate = collocate.lower() if lowercase else collocate
    rows = []

    for _, row in df.iterrows():
        tokens = tokenize(row.get(text_col), lowercase=lowercase, mode=tokenization_mode)
        for i, tok in enumerate(tokens):
            if tok != search_anchor:
                continue
            c_left = max(0, i - collocation_window)
            c_right = min(len(tokens), i + collocation_window + 1)
            local_window = tokens[c_left:i] + tokens[i + 1 : c_right]
            if search_collocate not in local_window:
                continue
            left_tokens = tokens[max(0, i - kwic_window) : i]
            right_tokens = tokens[i + 1 : min(len(tokens), i + kwic_window + 1)]
            rec = {
                "anchor": anchor,
                "collocate": collocate,
                "direction": row.get(direction_col),
                "left_context": " ".join(left_tokens),
                "node": tok,
                "right_context": " ".join(right_tokens),
                "full_text": row.get(text_col),
            }
            for col in metadata_cols:
                if col in row.index:
                    rec[col] = row.get(col)
            rows.append(rec)
            if max_examples is not None and len(rows) >= max_examples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def kwic_for_phrase(
    utterances_df: pd.DataFrame,
    phrase: str,
    text_col: str,
    direction_col: str = "direction",
    direction: str | None = None,
    kwic_window: int = 10,
    lowercase: bool = False,
    tokenization_mode: str = "regex",
    metadata_cols: Sequence[str] = ("conversation_id", "utterance_id", "speaker", "timestamp"),
    max_examples: int | None = 50,
) -> pd.DataFrame:
    """Return KWIC rows for a multi-token phrase such as 'Übergabe WE'."""
    df = utterances_df.copy()
    df[direction_col] = df[direction_col].map(normalize_direction)
    if direction is not None:
        df = df[df[direction_col].eq(normalize_direction(direction))].copy()

    phrase_tokens = tokenize(phrase, lowercase=lowercase, mode="whitespace")
    m = len(phrase_tokens)
    if m == 0:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        tokens = tokenize(row.get(text_col), lowercase=lowercase, mode=tokenization_mode)
        for i in range(0, len(tokens) - m + 1):
            if tokens[i : i + m] != phrase_tokens:
                continue
            left_tokens = tokens[max(0, i - kwic_window) : i]
            right_tokens = tokens[i + m : min(len(tokens), i + m + kwic_window)]
            rec = {
                "phrase": phrase,
                "direction": row.get(direction_col),
                "left_context": " ".join(left_tokens),
                "node": " ".join(tokens[i : i + m]),
                "right_context": " ".join(right_tokens),
                "full_text": row.get(text_col),
            }
            for col in metadata_cols:
                if col in row.index:
                    rec[col] = row.get(col)
            rows.append(rec)
            if max_examples is not None and len(rows) >= max_examples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6) Article-oriented condensation helpers
# -----------------------------------------------------------------------------


def summarize_for_article(
    colloc_df: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    """Create a compact profile table for manuscript interpretation.

    This is a quantitative scaffold. Add/adjust the final interpretation only after
    reviewing the KWIC evidence.
    """
    if colloc_df.empty:
        return pd.DataFrame(columns=[
            "Collocation profile",
            "Dominant modality",
            "Representative collocates",
            "Communicative interpretation",
            "Example items / KWIC evidence",
        ])

    profile_map = {
        "Patient anchoring around case references": ["PatName", "Hashtag_PatName"],
        "Handover and task-status bundles": ["Übergabe", "WE", "Todo", "kein_Todo", "kein"],
        "Therapy/group documentation and attendance": ["Rückmeldung", "Gruppe", "MT", "GT", "KT", "anwesend"],
        "Temporal coordination and scheduling": ["morgen", "heute", "Datum", "Uhr", "uhr", "WE"],
        "Clinical/procedural clarification": ["ÖGD", "Kolo", "BE", "Atorvastatin", "Suizidalität"],
        "Interactional work and addressivity": ["KolName", "Mention_KolName", "Hallo", "Hi", "danke", "bitte", "Guten"],
        "Documentation, communication, and resource logistics": ["Raum", "Brief", "Rückmeldung", "Link", "WIKI", "fertig"],
    }

    rows = []
    for profile, anchors in profile_map.items():
        sub = colloc_df[colloc_df["anchor"].isin(anchors)].copy()
        if sub.empty:
            rows.append({
                "Collocation profile": profile,
                "Dominant modality": "not observed / below threshold",
                "Representative collocates": "",
                "Communicative interpretation": "No collocates met the current threshold; consider lowering min_freq or checking KWIC for rare but relevant items.",
                "Example items / KWIC evidence": "To add after KWIC review",
            })
            continue

        modality_counts = sub.groupby("direction")["frequency"].sum().sort_values(ascending=False)
        if len(modality_counts) == 1:
            dominant = modality_counts.index[0]
        else:
            ratio = modality_counts.iloc[0] / max(modality_counts.iloc[1], 1)
            dominant = modality_counts.index[0] if ratio >= 1.25 else "mixed"

        top = (
            sub.sort_values(["frequency", "ll_g2", "association_score"], ascending=False)
            .head(top_n)
            .assign(pair=lambda x: x["anchor"] + "–" + x["collocate"] + " (" + x["direction"] + "/" + x["view"] + ", n=" + x["frequency"].astype(str) + ")")
        )
        rows.append({
            "Collocation profile": profile,
            "Dominant modality": dominant,
            "Representative collocates": "; ".join(top["pair"].tolist()),
            "Communicative interpretation": "Draft after KWIC validation; use collocates as evidence for the functional interpretation of this profile.",
            "Example items / KWIC evidence": "Insert 1-3 anonymized KWIC examples or item labels after manual review.",
        })
    return pd.DataFrame(rows)


def make_review_sample(
    colloc_df: pd.DataFrame,
    per_anchor: int = 10,
) -> pd.DataFrame:
    """Small review table: top collocates per anchor/direction/view."""
    if colloc_df.empty:
        return colloc_df
    return (
        colloc_df.sort_values(["anchor", "direction", "view", "frequency", "ll_g2"], ascending=[True, True, True, False, False])
        .groupby(["anchor", "direction", "view"], as_index=False, group_keys=False)
        .head(per_anchor)
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# 7) Example runner
# -----------------------------------------------------------------------------


def run_targeted_analysis(
    utterances_path: str | Path,
    output_dir: str | Path,
    direction_col: str = "direction",
    text_cols: dict[str, str] | None = None,
    keyness_xlsx: str | Path | None = None,
    use_core_items_from_keyness: bool = True,
    extra_anchors: Sequence[str] = DEFAULT_ANCHORS,
    window_size: int = 5,
    min_freq: int = 3,
    tokenization_mode: str = "regex",
) -> dict[str, pd.DataFrame]:
    """Run targeted collocations, generate KWIC samples, and export Excel files."""
    utterances_path = Path(utterances_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if utterances_path.suffix.lower() in {".xlsx", ".xls"}:
        utterances = pd.read_excel(utterances_path)
    else:
        utterances = pd.read_csv(utterances_path)

    anchors = set(extra_anchors)
    if keyness_xlsx is not None and use_core_items_from_keyness:
        anchors.update(load_core_items_as_anchors(keyness_xlsx))

    config = CollocationConfig(
        direction_col=direction_col,
        text_cols=text_cols,
        anchors=tuple(sorted(anchors)),
        window_size=window_size,
        min_freq=min_freq,
        tokenization_mode=tokenization_mode,
    )

    colloc = targeted_collocations(utterances, config)
    review = make_review_sample(colloc, per_anchor=10)
    article = summarize_for_article(colloc, top_n=8)

    # KWIC examples for the strongest patterns in the review table.
    kwic_rows = []
    if not review.empty:
        for _, r in review.head(80).iterrows():
            text_col = config.resolved_text_cols()[r["view"]]
            examples = kwic_for_anchor_collocate(
                utterances,
                anchor=r["anchor"],
                collocate=r["collocate"],
                text_col=text_col,
                direction_col=direction_col,
                direction=r["direction"],
                collocation_window=window_size,
                kwic_window=10,
                max_examples=3,
            )
            if not examples.empty:
                examples.insert(0, "view", r["view"])
                examples.insert(0, "frequency", r["frequency"])
                examples.insert(0, "association_score", r["association_score"])
                kwic_rows.append(examples)
    kwic = pd.concat(kwic_rows, ignore_index=True) if kwic_rows else pd.DataFrame()

    xlsx_path = output_dir / "targeted_collocations_kwic_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        colloc.to_excel(writer, sheet_name="collocations_long", index=False)
        review.to_excel(writer, sheet_name="review_top_collocates", index=False)
        kwic.to_excel(writer, sheet_name="kwic_examples", index=False)
        article.to_excel(writer, sheet_name="article_profile_draft", index=False)
        pd.DataFrame(ARTICLE_PROFILE_SEEDS).to_excel(writer, sheet_name="article_profile_template", index=False)

    print(f"Saved: {xlsx_path}")
    return {
        "collocations_long": colloc,
        "review_top_collocates": review,
        "kwic_examples": kwic,
        "article_profile_draft": article,
    }



# -----------------------------------------------------------------------------
# 8) CLI runner adapted to the current project structure
# -----------------------------------------------------------------------------

def _write_outputs(
    output_dir: Path,
    colloc: pd.DataFrame,
    review: pd.DataFrame,
    kwic: pd.DataFrame,
    article: pd.DataFrame,
) -> None:
    """
    Write main results to Excel. If Excel writing fails, fall back to CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = output_dir / "targeted_collocations_kwic_results_clean_lexical.xlsx"

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            colloc.to_excel(writer, sheet_name="collocations_long", index=False)
            review.to_excel(writer, sheet_name="review_top_collocates", index=False)
            kwic.to_excel(writer, sheet_name="kwic_examples", index=False)
            article.to_excel(writer, sheet_name="article_profile_draft", index=False)
            pd.DataFrame(ARTICLE_PROFILE_SEEDS).to_excel(
                writer,
                sheet_name="article_profile_template",
                index=False,
            )
        print(f"\nSaved Excel result file:\n{xlsx_path.resolve()}")
        print(f"File size: {xlsx_path.stat().st_size:,} bytes")
        return
    except Exception as exc:
        print("\nCould not write Excel workbook.")
        print(f"Reason: {exc}")
        print("Writing CSV fallback files instead.")

    csv_dir = output_dir / "targeted_collocations_kwic_results_clean_lexical_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    colloc.to_csv(csv_dir / "collocations_long.csv", index=False, encoding="utf-8-sig")
    review.to_csv(csv_dir / "review_top_collocates.csv", index=False, encoding="utf-8-sig")
    kwic.to_csv(csv_dir / "kwic_examples.csv", index=False, encoding="utf-8-sig")
    article.to_csv(csv_dir / "article_profile_draft.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ARTICLE_PROFILE_SEEDS).to_csv(
        csv_dir / "article_profile_template.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved CSV fallback folder:\n{csv_dir.resolve()}")


def _find_default_keyness_file(project_dir: Path) -> Path | None:
    """
    Try common locations for the condensed JMIR keyness table.
    Returns None if no file is found.
    """
    candidates = [
        project_dir / "outputs/results/tables/keyness_condensed_results_table_JMIR_updated_SJ.xlsx",
        project_dir / "outputs/results/keyness_condensed_results_table_JMIR_updated_SJ.xlsx",
        project_dir / "output/results/tables/keyness_condensed_results_table_JMIR_updated_SJ.xlsx",
        project_dir / "keyness_condensed_results_table_JMIR_updated_SJ.xlsx",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted collocation and KWIC analysis on the combined cleaned utterance table."
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory. Default: current working directory.",
    )

    parser.add_argument(
        "--utterances",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/utterances_for_collocation_clean_lexical.csv"),
        help="Combined utterance CSV relative to project dir or absolute.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/results/collocations"),
        help="Output directory relative to project dir or absolute.",
    )

    parser.add_argument(
        "--keyness-xlsx",
        type=Path,
        default=None,
        help="Optional condensed JMIR keyness table. If omitted, common default locations are checked.",
    )

    parser.add_argument(
        "--no-keyness-anchors",
        action="store_true",
        help="Do not add core-result items from the condensed keyness table.",
    )

    parser.add_argument(
        "--content-col",
        type=str,
        default="content_text",
        help="Column for the content view. Default: content_text.",
    )

    parser.add_argument(
        "--interaction-col",
        type=str,
        default="interaction_text",
        help="Column for the interaction view. Default: interaction_text.",
    )

    parser.add_argument(
        "--include-interaction-view",
        action="store_true",
        help="Also analyze interaction_text. Use only if it is a genuinely separate interaction view.",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Collocation window size left/right. Default: 5.",
    )

    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Minimum co-occurrence frequency. Default: 3.",
    )

    parser.add_argument(
        "--max-kwic-patterns",
        type=int,
        default=80,
        help="How many top collocation patterns to sample for KWIC examples. Default: 80.",
    )

    parser.add_argument(
        "--kwic-per-pattern",
        type=int,
        default=3,
        help="Maximum KWIC examples per collocation pattern. Default: 3.",
    )

    return parser.parse_args()


def resolve_cli_path(path: Path, project_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return project_dir / path


def main_cli() -> int:
    args = parse_cli_args()

    project_dir = args.project_dir.resolve()
    utterances_path = resolve_cli_path(args.utterances, project_dir)
    output_dir = resolve_cli_path(args.out_dir, project_dir)

    if args.keyness_xlsx is not None:
        keyness_xlsx = resolve_cli_path(args.keyness_xlsx, project_dir)
    else:
        keyness_xlsx = _find_default_keyness_file(project_dir)

    print("Project directory:")
    print(project_dir)

    print("\nInput/output paths:")
    print(f"Utterances: {utterances_path}")
    print(f"Output dir: {output_dir}")
    print(f"Keyness:    {keyness_xlsx if keyness_xlsx else 'not used / not found'}")

    if not utterances_path.exists():
        print("\nERROR:")
        print(f"Utterance table not found: {utterances_path}")
        return 1

    print("\nLoading combined utterance table...")
    utterances = pd.read_csv(utterances_path)
    print(f"Utterance shape: {utterances.shape}")
    print("Columns:")
    print(utterances.columns.tolist())

    required_cols = ["direction", args.content_col]
    missing = [col for col in required_cols if col not in utterances.columns]
    if missing:
        print("\nERROR:")
        print(f"Missing required columns: {missing}")
        return 1

    text_cols = {"content": args.content_col}

    if args.include_interaction_view:
        if args.interaction_col not in utterances.columns:
            print("\nERROR:")
            print(f"Interaction column not found: {args.interaction_col}")
            return 1
        text_cols["interaction"] = args.interaction_col
    else:
        print("\nInteraction view is not analyzed in this run.")
        print("Reason: content_text and interaction_text are currently identical unless separate view-specific stopwording has been applied.")

    anchors = set(DEFAULT_ANCHORS)

    if keyness_xlsx is not None and keyness_xlsx.exists() and not args.no_keyness_anchors:
        try:
            keyness_anchors = load_core_items_as_anchors(keyness_xlsx)
            anchors.update(keyness_anchors)
            print(f"\nAdded anchors from keyness table: {len(keyness_anchors)}")
        except Exception as exc:
            print("\nCould not load anchors from keyness table.")
            print(f"Reason: {exc}")
            print("Continuing with DEFAULT_ANCHORS only.")
    else:
        print("\nUsing DEFAULT_ANCHORS only.")

    print(f"Total anchors used: {len(anchors)}")
    print("Anchor preview:")
    print(sorted(anchors)[:50])

    config = CollocationConfig(
        direction_col="direction",
        text_cols=text_cols,
        anchors=tuple(sorted(anchors)),
        window_size=args.window_size,
        min_freq=args.min_freq,
        tokenization_mode="regex",
        lowercase=False,
    )

    print("\nRunning targeted collocation analysis...")
    colloc = targeted_collocations(utterances, config)
    print(f"Collocations found: {len(colloc)}")

    if colloc.empty:
        print("\nNo collocations met the current threshold.")
        print("Try --min-freq 2 for diagnostic purposes, but keep min_freq >= 3 for final reporting if possible.")
        return 0

    review = make_review_sample(colloc, per_anchor=10)
    article = summarize_for_article(colloc, top_n=8)

    print(f"Review rows: {len(review)}")
    print(f"Article profile rows: {len(article)}")

    print("\nGenerating KWIC examples for top patterns...")
    kwic_rows = []

    for _, r in review.head(args.max_kwic_patterns).iterrows():
        text_col = config.resolved_text_cols()[r["view"]]

        examples = kwic_for_anchor_collocate(
            utterances,
            anchor=r["anchor"],
            collocate=r["collocate"],
            text_col=text_col,
            direction_col="direction",
            direction=r["direction"],
            collocation_window=args.window_size,
            kwic_window=10,
            max_examples=args.kwic_per_pattern,
        )

        if not examples.empty:
            examples.insert(0, "view", r["view"])
            examples.insert(0, "frequency", r["frequency"])
            examples.insert(0, "association_score", r["association_score"])
            examples.insert(0, "ll_g2", r["ll_g2"])
            kwic_rows.append(examples)

    kwic = pd.concat(kwic_rows, ignore_index=True) if kwic_rows else pd.DataFrame()
    print(f"KWIC rows: {len(kwic)}")

    _write_outputs(
        output_dir=output_dir,
        colloc=colloc,
        review=review,
        kwic=kwic,
        article=article,
    )

    print("\nTop 20 review patterns:")
    display_cols = [
        "anchor",
        "collocate",
        "direction",
        "view",
        "frequency",
        "association_score",
        "ll_g2",
        "possible_category",
    ]
    display_cols = [c for c in display_cols if c in review.columns]
    print(review[display_cols].head(20).to_string(index=False))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    import argparse
    sys.exit(main_cli())
