"""
cleaning.py

Rule-based text cleaning functions for lexical corpus analyses.

This module contains the actual cleaning logic used by
scripts/03_clean_lexical.py.

The cleaning rules are designed for lexical frequency analyses,
N-gram analyses, and related corpus-linguistic analyses.

The original corpus tables should remain unchanged.

to run use: 
python scripts/03_clean_lexical.py --corpus both
python scripts/06_frequency_ngram_keyness.py --corpus both --min-count 3

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
    Normalize gender-inclusive clinical, patient-related, and role terms.

    The mapping is corpus-specific and based on manual validation of
    original tokens preceding :innen or *innen.

    Examples:
    Patient*innen              -> Patient
    Patient*Innen              -> Patient
    Probatorik-Patient*innen   -> Probatorik_Patient
    MitpatientInnen            -> Mitpatient
    Therapeut:innen            -> Therapeut
    Ärzt:innen                 -> Arzt
    Praktikant*innen           -> KolName
    Freund:innen               -> Freund
    """

    gender_role_map = {
        # Patient-related terms
        "patient": "Patient",
        "pattien": "Patient",
        "probatorik-patient": "Probatorik_Patient",
        "probatorikpatient": "Probatorik_Patient",
        "mitpatient": "Mitpatient",
        "zimmernachbar": "Mitpatient",

        # Colleague / internal team references
        "kolleg": "KolName",
        "praktikant": "KolName",
        "psychologiepraktikant": "KolName",
        "assistent": "KolName",
        "mitarbeiter": "KolName",

        # Clinical roles
        "therapeut": "Therapeut",
        "bezugstherapeut": "Bezugstherapeut",
        "kreativtherapeut": "Kreativtherapeut",
        "psychotherapeut": "Psychotherapeut",
        "behandler": "Behandler",
        "ärzt": "Arzt",
        "arzt": "Arzt",
        "ärtz": "Arzt",
        "psycholog": "Psychologe",
        "psychiater": "Psychiater",

        # Social roles
        "freund": "Freund",

        # Project / research / other roles
        "interessent": "Interessent",
        "wissenschaftler": "Wissenschaftler",
        "forscher": "Forscher",
        "deutschmuttersprachler": "Deutschmuttersprachler",
    }

    for base, replacement in gender_role_map.items():
        escaped_base = re.escape(base)

        # Forms with colon, asterisk, slash, or underscore:
        # Patient:innen, Patient*innen, Patient_innen, Patient/innen
        text = re.sub(
            rf"\b{escaped_base}\s*[:*/_]\s*[Ii]nnen\b",
            f" {replacement} ",
            text,
            flags=re.IGNORECASE,
        )

        # Already split forms:
        # Patient innen
        text = re.sub(
            rf"\b{escaped_base}\s+[Ii]nnen\b",
            f" {replacement} ",
            text,
            flags=re.IGNORECASE,
        )

        # CamelCase / Binnen-I forms:
        # PatientInnen
        text = re.sub(
            rf"\b{escaped_base}[Ii]nnen\b",
            f" {replacement} ",
            text,
            flags=re.IGNORECASE,
        )

    return text

def normalize_patient_role_variants(text: str) -> str:
    """
    Normalize ordinary patient-related lexical variants.

    This function handles non-gender-marker variants that are not captured
    by the :innen / *innen normalization rules.

    Examples:
    Patientin        -> Patient
    Patienten        -> Patient
    PAatientin       -> Patient
    Mitpatienten     -> Mitpatient
    Mitpatientin      -> Mitpatient
    Schmerzpatienten -> Schmerzpatient
    """

    # Typo observed in the corpus
    text = re.sub(
        r"\bPAatient(?:in|innen|en)?\b",
        " Patient ",
        text,
        flags=re.IGNORECASE,
    )

    # Patient / Patientin / Patienten / Patientinnen
    text = re.sub(
        r"\bPatient(?:in|innen|en)?\b",
        " Patient ",
        text,
        flags=re.IGNORECASE,
    )

    # Mitpatient / Mitpatientin / Mitpatienten / Mitpatientinnen
    text = re.sub(
        r"\bMitpatient(?:in|innen|en)?\b",
        " Mitpatient ",
        text,
        flags=re.IGNORECASE,
    )

    # Schmerzpatient / Schmerzpatientin / Schmerzpatienten / Schmerzpatientinnen
    text = re.sub(
        r"\bSchmerzpatient(?:in|innen|en)?\b",
        " Schmerzpatient ",
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
    cleaned = normalize_patient_role_variants(cleaned)

    # ------------------------------------------------------------------
    # 0c. Normalize gender-inclusive colleague/user references
    # ------------------------------------------------------------------

    cleaned = normalize_gender_colleague_user_references(cleaned)

    # ------------------------------------------------------------------
    # 0d. Harmonize common clinical abbreviations
    # ------------------------------------------------------------------

    # Pat / Pat. is used as an abbreviation for patient(s) in this corpus.
    # It is standardized to Patient to avoid artificial token splitting
    # between Pat and Patient.
    cleaned = re.sub(
        r"\bPat\.?\b",
        " Patient ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 0e. Normalize URL remnants
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"\bhttps?://\S+",
        " Link ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bwww\.\S+",
        " Link ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 10b. Preserve question marks as interaction markers
    # ------------------------------------------------------------------

    cleaned = re.sub(
        r"\?+",
        " QuestionMark ",
        cleaned,
    )

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

    cleaned = re.sub(r"[<>@#,:;\*\(\)\.\"'„“‚‘’`´\!]", " ", cleaned)
    #cleaned = re.sub(r"[<>@#,:;\*\(\)\.\"'„“‚‘’`´]", " ", cleaned)

    # Hyphens are removed only after meaningful compounds have been handled
    cleaned = cleaned.replace("-", " ")

    # ------------------------------------------------------------------
    # 12. Normalize whitespace
    # ------------------------------------------------------------------

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned