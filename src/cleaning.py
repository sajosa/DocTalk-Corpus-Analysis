"""
cleaning.py

Rule-based text cleaning functions for lexical corpus analyses.

This module contains the actual cleaning logic used by
scripts/03_clean_lexical.py.

The cleaning rules are designed for lexical frequency analyses,
N-gram analyses, and related corpus-linguistic analyses.

The original corpus tables should remain unchanged.
"""

import re


def _normalize_hashtag_content(hashtag: str) -> str:
    """
    Normalize the content of a hashtag while preserving it as one token.
    """
    hashtag = hashtag.strip()
    hashtag = hashtag.replace("-", "_")
    hashtag = re.sub(r"[^\wÄÖÜäöüß_]", "", hashtag)
    return hashtag


def clean_text_lexical(text: str) -> str:
    """
    Apply rule-based lexical cleaning to one message text.
    """

    if not isinstance(text, str):
        return ""

    cleaned = text

    # ------------------------------------------------------------------
    # 1. Standardize functional communication markers
    # ------------------------------------------------------------------

    # Group-wide mentions
    cleaned = re.sub(
        r"@(all|channel|here)\b",
        " Mention_All ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Mentions of anonymized full names, if present
    cleaned = re.sub(
        r"@<Vorname>\s*<Nachname>",
        " Mention_KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Mentions of anonymized colleague surnames
    cleaned = re.sub(
        r"@<Nachname>",
        " Mention_KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Mentions of anonymized colleagues
    cleaned = re.sub(
        r"@<Person\d+>",
        " Mention_KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Mentions of anonymized first names, if present
    cleaned = re.sub(
        r"@<Vorname>",
        " Mention_KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Mentions written as @Vorname in the anonymized corpus
    cleaned = re.sub(
        r"@Vorname\b",
        " Mention_KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Patient-related hashtags
    cleaned = re.sub(
        r"#<Nachname>",
        " Hashtag_PatName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Thematic hashtags
    cleaned = re.sub(
        r"#([\wÄÖÜäöüß-]+)",
        lambda m: " Hashtag_" + _normalize_hashtag_content(m.group(1)) + " ",
        cleaned,
    )

    # ------------------------------------------------------------------
    # 2. Remove export artefacts
    # ------------------------------------------------------------------

    cleaned = re.sub(r"<\s*enter\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\benter>", " ", cleaned, flags=re.IGNORECASE)

    # ------------------------------------------------------------------
    # 3. Prevent artificial token duplication
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"\bStation\s+<Station>",
        " Station ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bZi\.?\s+<Zimmernummer>",
        " Zi ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bZimmer\s+<Zimmernummer>",
        " Zimmer ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 4. Harmonize patient-name and full-name placeholders
    # ------------------------------------------------------------------

    # Salutation + surname indicates a patient-name reference
    cleaned = re.sub(
        r"\b(Frau|Fr\.?|Herr|Hr\.?|Herrn)\s+<Nachname>",
        " PatName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Full anonymized names without salutation refer to colleagues in this corpus
    cleaned = re.sub(
        r"<Vorname>\s*<Nachname>",
        " KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remaining anonymized surnames are treated as patient-name references
    cleaned = re.sub(
        r"<Nachname>",
        " PatName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"<Aussprache_Nachname>",
        " Aussprache_PatName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"<Nachname_Aussprache>",
        " Aussprache_PatName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"<Name_Aussprache>",
        " Aussprache_PatName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 5. Remove non-informative placeholders
    # ------------------------------------------------------------------

    remove_placeholders = [
        "Zimmernummer",
        "Station",
        "Stationsteam",
        "Alter",
        "BMI",
    ]

    for placeholder in remove_placeholders:
        cleaned = re.sub(
            rf"<{placeholder}>",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # 6. Harmonize colleague-related placeholders
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"<Person\d+>",
        " KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"<Vorname>",
        " KolName ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 7. Standardize informative anonymization placeholders
    # ------------------------------------------------------------------

    replacement_map = {
        "<Datum>": "Datum",
        "<Monat>": "Monat",
        "<Jahr>": "Jahr",
        "<Klinik>": "Klinik",
        "<Klinik1>": "Klinik",
        "<Klink>": "Klinik",
        "<Klinikstandort>": "Klinikstandort",
        "<Telefonnummer>": "Telefonnummer",
        "<Telefonnumemr>": "Telefonnummer",
        "<Dateipfad>": "Link_intern",
        "<Link>": "Link",
        "<Ort>": "Ort",
    }

    for old, new in replacement_map.items():
        cleaned = re.sub(
            re.escape(old),
            f" {new} ",
            cleaned,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # 8. Harmonize ToDo expressions and preserve negations
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*to[\s-]?dos?\b",
        " kein_Todo ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*todos?\b",
        " kein_Todo ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bto[\s-]?dos?\b|\btodo'?s?\b|\btodos\b",
        " Todo ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 9. Remove non-informative salutations
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"\b(Frau|Fr|Herr|Hr|Herrn)\b\.?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 10. Harmonize corpus-specific terms
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"\bGesprächsgruppe\b",
        " GT ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bkt\b",
        " KT ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bmt\b",
        " MT ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 11. Remove residual punctuation and technical symbols
    # ------------------------------------------------------------------

    cleaned = re.sub(r"[<>@#,:;\*\(\)\.]", " ", cleaned)

    # Hyphens are removed only after meaningful compounds have been handled
    cleaned = cleaned.replace("-", " ")

    # ------------------------------------------------------------------
    # 12. Normalize whitespace
    # ------------------------------------------------------------------

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned