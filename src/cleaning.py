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


# ---------------------------------------------------------------------
# Gender-inclusive forms and corpus-specific compounds
# ---------------------------------------------------------------------

def normalize_gender_specific_compounds(text: str) -> str:
    """
    Normalize selected German gender-inclusive compounds before
    general punctuation removal.

    These rules are corpus-specific and based on manual validation
    of token and N-gram frequency tables.

    Examples:
    Benutzer:inname          -> KolName
    Benutzer inname          -> KolName
    Nutzer:innentreffen      -> Projekt_Nutzertreffen
    Nutzer innentreffen      -> Projekt_Nutzertreffen
    Behandler:innenwechsel   -> Behandlerwechsel
    Behandler innenwechsel   -> Behandlerwechsel
    """

    # Known artefact from Benutzer:innen / Benutzer:inname.
    # In this corpus this refers to platform users / colleagues.
    text = re.sub(
        r"\bBenutzer\s*[:*/_]\s*inname\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bBenutzer\s+inname\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Project-related user meeting format.
    # Not clinically content-relevant; can later be removed in the
    # content-token view via stopwords.
    text = re.sub(
        r"\bNutzer\s*[:*/_]\s*innentreffen\b",
        " Projekt_Nutzertreffen ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bNutzer\s+innentreffen\b",
        " Projekt_Nutzertreffen ",
        text,
        flags=re.IGNORECASE,
    )

    # Clinically relevant provider change.
    # This should remain available for content analyses.
    text = re.sub(
        r"\bBehandler\s*[:*/_]\s*innenwechsel\b",
        " Behandlerwechsel ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bBehandler\s+innenwechsel\b",
        " Behandlerwechsel ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def normalize_gender_clinical_role_terms(text: str) -> str:
    """
    Normalize gender-inclusive clinical role and patient terms.

    These rules preserve clinically or organizationally meaningful
    categories while preventing artificial token splitting.

    Examples:
    Bezugstherapeut:innen       -> Bezugstherapeut
    Therapeut*innen             -> Therapeut
    Patient*innen               -> Patient
    MitpatientInnen             -> Mitpatient
    Probatorik-Patient*innen    -> Probatorik_Patient
    Ärzt:innen                  -> Arzt
    Ärtz:innen                  -> Arzt
    """

    # Bezugstherapeut:innen / Bezugstherapeut*innen / BezugstherapeutInnen
    text = re.sub(
        r"\bBezugstherapeut\s*[:*/_]\s*innen\b",
        " Bezugstherapeut ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bBezugstherapeutInnen\b",
        " Bezugstherapeut ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bBezugstherapeut\s+innen\b",
        " Bezugstherapeut ",
        text,
        flags=re.IGNORECASE,
    )

    # Therapeut:innen / Therapeut*innen / TherapeutInnen
    text = re.sub(
        r"\bTherapeut\s*[:*/_]\s*innen\b",
        " Therapeut ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bTherapeutInnen\b",
        " Therapeut ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bTherapeut\s+innen\b",
        " Therapeut ",
        text,
        flags=re.IGNORECASE,
    )

    # Praktikant*innen / Praktikant:innen / PraktikantInnen
    # In this corpus, these are team/member references.
    text = re.sub(
        r"\bPraktikant\s*[:*/_]\s*innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bPraktikantInnen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bPraktikant\s+innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Patient*innen / Patient:innen / PatientInnen / Patient*Innen
    text = re.sub(
        r"\bPatient\s*[:*/_]\s*innen\b",
        " Patient ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bPatientInnen\b",
        " Patient ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bPatient\s+innen\b",
        " Patient ",
        text,
        flags=re.IGNORECASE,
    )

    # Mitpatient innen / Mitpatient:innen / Mitpatient*innen / MitpatientInnen
    text = re.sub(
        r"\bMitpatient\s*[:*/_]\s*innen\b",
        " Mitpatient ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bMitpatientInnen\b",
        " Mitpatient ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bMitpatient\s+innen\b",
        " Mitpatient ",
        text,
        flags=re.IGNORECASE,
    )

    # Probatorik-Patient*innen / Probatorik Patient innen / Probatorik-PatientInnen
    text = re.sub(
        r"\bProbatorik[-\s]+Patient\s*[:*/_]\s*innen\b",
        " Probatorik_Patient ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bProbatorik[-\s]+PatientInnen\b",
        " Probatorik_Patient ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bProbatorik[-\s]+Patient\s+innen\b",
        " Probatorik_Patient ",
        text,
        flags=re.IGNORECASE,
    )

    # Ärzt:innen / Arzt:innen / Ärtz:innen / Ärzt*innen / Arzt*innen / ÄrztInnen
    text = re.sub(
        r"\b(Ärzt|Arzt|Ärtz)\s*[:*/_]\s*innen\b",
        " Arzt ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(Ärzt|Arzt|Ärtz)\s+innen\b",
        " Arzt ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bÄrztInnen\b|\bArztInnen\b|\bÄrtzInnen\b",
        " Arzt ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def normalize_gender_colleague_user_references(text: str) -> str:
    """
    Normalize gender-inclusive colleague/user references to KolName.

    In this corpus, forms such as Kolleg:innen, Benutzer:innen, Nutzer:innen
    refer to colleagues or platform users within the clinical team.

    Examples:
    Kolleg:innen       -> KolName
    Kolleg*innen       -> KolName
    Kolleg_innen       -> KolName
    Kolleg innen       -> KolName
    Benutzer:innen     -> KolName
    Benutzer innen     -> KolName
    Nutzer:innen       -> KolName
    Nutzer innen       -> KolName
    """

    # Kolleg:innen / Kolleg*innen / Kolleg_innen / Kolleg/innen
    text = re.sub(
        r"\bKolleg(?:e|en|in|innen)?\s*[:*/_]\s*innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Kolleg innen / Kollegen innen / Kolleginnen innen
    text = re.sub(
        r"\bKolleg(?:e|en|in|innen)?\s+innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Benutzer:innen / Benutzer*innen / Benutzer_innen / Benutzer/innen
    text = re.sub(
        r"\bBenutzer\s*[:*/_]\s*innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Benutzer innen
    text = re.sub(
        r"\bBenutzer\s+innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Nutzer:innen / Nutzer*innen / Nutzer_innen / Nutzer/innen
    text = re.sub(
        r"\bNutzer\s*[:*/_]\s*innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    # Nutzer innen
    text = re.sub(
        r"\bNutzer\s+innen\b",
        " KolName ",
        text,
        flags=re.IGNORECASE,
    )

    return text


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _normalize_hashtag_content(hashtag: str) -> str:
    """
    Normalize the content of a hashtag while preserving it as one token.
    """

    hashtag = hashtag.strip()
    hashtag = hashtag.replace("-", "_")
    hashtag = re.sub(r"[^\wÄÖÜäöüß_]", "", hashtag)

    return hashtag


# ---------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------

def clean_text_lexical(text: str) -> str:
    """
    Apply rule-based lexical cleaning to one message text.
    """

    if not isinstance(text, str):
        return ""

    cleaned = text

    # ------------------------------------------------------------------
    # 0a. Normalize selected gender-inclusive compounds first
    # ------------------------------------------------------------------

    cleaned = normalize_gender_specific_compounds(cleaned)

    # ------------------------------------------------------------------
    # 0b. Normalize gender-inclusive clinical role and patient terms
    # ------------------------------------------------------------------

    cleaned = normalize_gender_clinical_role_terms(cleaned)

    # ------------------------------------------------------------------
    # 0c. Normalize gender-inclusive colleague/user references
    # ------------------------------------------------------------------

    cleaned = normalize_gender_colleague_user_references(cleaned)

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
        r"<Name_Abkürzung>",
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

    # Remove possessive/apostrophe remnants after standardized negated ToDo tokens
    cleaned = re.sub(
        r"\bkein_Todo\s*['’`´]?\s*s\b",
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

    cleaned = re.sub(r"[<>@#,:;\*\(\)\.\"'„“‚‘’`´]", " ", cleaned)

    # Hyphens are removed only after meaningful compounds have been handled
    cleaned = cleaned.replace("-", " ")

    # ------------------------------------------------------------------
    # 12. Normalize whitespace
    # ------------------------------------------------------------------

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned