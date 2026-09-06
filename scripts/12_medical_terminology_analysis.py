#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_medical_terminology_analysis.py

Model-supported extraction of candidate clinical entity mentions in the DocTalk corpus.

Purpose
-------
This script provides an open-source model-supported extraction of candidate clinical entity mentions,
including diagnoses, symptoms, medications, procedures/tests and treatments, from German clinical chat messages.

The script deliberately treats all model outputs as CANDIDATES requiring manual
review. It does not infer disease prevalence or treatment prevalence among patients.
It describes what was mentioned in communication.

Recommended first run
---------------------
python scripts/12_medical_terminology_analysis.py \
  --corpus both \
  --backend humadex_medner \
  --min_score 0.50

Install dependencies
--------------------
pip install pandas openpyxl tqdm transformers torch

Optional for GLiNER zero-shot NER:
pip install gliner

Optional GLiNER run:
python scripts/12_medical_terminology_analysis.py \
  --corpus both \
  --backend gliner \
  --gliner_threshold 0.35

Backends
--------
1) humadex
   HUMADEX/german_medical_ner
   Broad German medical NER with labels PROBLEM, TEST, TREATMENT.
   Useful to find disease/symptom/problem, diagnostic tests/procedures and treatments.

2) medner
   pei-germany/MEDNER-de-fp-gbert
   German medicinal product recognizer. Useful as medication-focused second pass.

3) gliner
   VAGOsolutions/SauerkrautLM-GLiNER (or another GLiNER model)
   Prompted / label-list NER. Useful as exploratory cross-check with labels such as
   diagnosis, symptom, medication, medical procedure, clinical risk.

Outputs
-------
outputs/confidential/review_files/medical_terminology_model/
  model_entity_mentions_long.xlsx
  model_entity_candidates_for_review.xlsx
  model_entity_summary_by_corpus.xlsx
  model_entity_summary_by_type.xlsx
  model_run_config.json

Notes
-----
- The first run downloads open-source models from Hugging Face. Inference is local.
- No message text is intentionally sent to an external API by this script.
- Outputs may contain original message snippets or full messages if --include_full_message is used and must remain confidential.
- Chat messages are short, elliptical, anonymized and domain-specific; therefore manual
  validation is essential.
- For publication, report this as model-supported candidate extraction with manual review,
  not as fully automated clinical NER.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger("medical_terminology_model")


DEFAULT_DIRECT_PATH = "outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv"
DEFAULT_GROUP_PATH = "outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv"
DEFAULT_OUTPUT_DIR = "outputs/confidential/review_files/medical_terminology_model"

DEFAULT_HUMADEX_MODEL = "HUMADEX/german_medical_ner"
DEFAULT_MEDNER_MODEL = "pei-germany/MEDNER-de-fp-gbert"
DEFAULT_GLINER_MODEL = "VAGOsolutions/SauerkrautLM-GLiNER"

DEFAULT_GLINER_LABELS = [
    "diagnosis",
    "symptom",
    "medication",
    "medical procedure",
    "diagnostic test",
    "clinical risk",
    "therapy",
]

STANDARD_TYPE_MAP = {
    # HUMADEX
    "PROBLEM": "diagnosis_or_symptom",
    "TEST": "diagnostic_test_or_procedure",
    "TREATMENT": "treatment_or_medication",
    # GERNERMED / medication-style models
    "DRUG": "medication",
    "Drug": "medication",
    "MEDICINAL_PRODUCT": "medication",
    "MEDICATION": "medication",
    "VACCINE": "medication_or_vaccine",
    "STRENGTH": "medication_strength",
    "Strength": "medication_strength",
    "DOSAGE": "medication_dosage",
    "Dosage": "medication_dosage",
    "FREQUENCY": "medication_frequency",
    "Frequency": "medication_frequency",
    "DURATION": "medication_duration",
    "Duration": "medication_duration",
    "FORM": "medication_form",
    "Form": "medication_form",
    "ROUTE": "medication_route",
    "Route": "medication_route",
    # GLiNER prompted labels
    "diagnosis": "diagnosis",
    "symptom": "symptom",
    "medication": "medication",
    "medical procedure": "procedure",
    "diagnostic test": "diagnostic_test_or_procedure",
    "clinical risk": "clinical_risk",
    "therapy": "therapy_or_treatment",
}

# Terms that are important in prior analyses but should not be interpreted as clinical entities here.
DEFAULT_EXCLUSION_NORMALIZED = {
    "patname",
    "hashtag_patname",
    "kolname",
    "mention_kolname",
    "übergabe",
    "rückmeldung",
    "we",
    "wochenende",
    "kein_todo",
    "todo",
    "raum",
    "station",
    "datum",
    "telefonnummer",
    "link",
    "link_intern",
    "klinik",
    "klinikstandort",
    "gruppe",  # single token only; MT_Gruppe etc. may still be useful via GLiNER/HUMADEX if detected in context
}


@dataclass
class CorpusSpec:
    corpus_type: str
    path: Path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open-source model-supported extraction of candidate clinical entity "
            "mentions in German clinical chat messages."
        )
    )
    parser.add_argument("--project_root", default=".", help="Project root directory. Default: current directory.")
    parser.add_argument("--direct_path", default=DEFAULT_DIRECT_PATH, help="Path to cleaned direct-message CSV.")
    parser.add_argument("--group_path", default=DEFAULT_GROUP_PATH, help="Path to cleaned group-message CSV.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--corpus", choices=["direct", "group", "both"], default="both")
    parser.add_argument("--text_column", default="text_clean_lexical", help="Column used for extraction.")
    parser.add_argument("--original_text_column", default="text_original", help="Column used for readable contexts if present.")
    parser.add_argument(
        "--backend",
        choices=["all", "humadex", "medner", "gliner", "humadex_medner"],
        default="all",
        help="Which open-source NER backend(s) to run.",
    )
    parser.add_argument("--humadex_model", default=DEFAULT_HUMADEX_MODEL)
    parser.add_argument("--medner_model", default=DEFAULT_MEDNER_MODEL)
    parser.add_argument("--gliner_model", default=DEFAULT_GLINER_MODEL)
    parser.add_argument(
        "--gliner_labels",
        default=",".join(DEFAULT_GLINER_LABELS),
        help="Comma-separated labels for GLiNER prompted NER.",
    )
    parser.add_argument("--min_score", type=float, default=0.50, help="Minimum model confidence score to retain.")
    parser.add_argument("--gliner_threshold", type=float, default=0.35, help="GLiNER threshold.")
    parser.add_argument("--min_occurrences", type=int, default=1, help="Minimum total occurrence count for summary review table.")
    parser.add_argument("--min_message_count", type=int, default=1, help="Minimum message count for summary review table.")
    parser.add_argument("--context_chars", type=int, default=90, help="Characters left/right around entity for context.")
    parser.add_argument("--max_kwic_per_candidate", type=int, default=10, help="Max contexts per normalized candidate.")
    parser.add_argument("--include_full_message", action="store_true", help="Include full original/cleaned message text in long output.")
    parser.add_argument("--write_csv", action="store_true", help="Also write CSV versions of main outputs.")
    parser.add_argument("--max_messages", type=int, default=None, help="Optional limit per corpus for testing.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Inference device.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(project_root: Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else project_root / p


def get_corpus_specs(args: argparse.Namespace, project_root: Path) -> List[CorpusSpec]:
    specs: List[CorpusSpec] = []
    if args.corpus in {"direct", "both"}:
        specs.append(CorpusSpec("direct", resolve_path(project_root, args.direct_path)))
    if args.corpus in {"group", "both"}:
        specs.append(CorpusSpec("group", resolve_path(project_root, args.group_path)))
    return specs


def load_corpus(spec: CorpusSpec, text_column: str, original_text_column: str, max_messages: Optional[int]) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"Input file not found for {spec.corpus_type}: {spec.path}")
    df = pd.read_csv(spec.path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {spec.path}. Available columns: {list(df.columns)}")
    if original_text_column not in df.columns:
        LOGGER.warning("Original text column '%s' not found in %s. Falling back to extraction text.", original_text_column, spec.path)
        df[original_text_column] = df[text_column]
    df = df.copy()
    df["corpus_type"] = spec.corpus_type
    if "id" not in df.columns:
        df["id"] = [f"{spec.corpus_type}_{i}" for i in range(len(df))]
    if "conversation_id" not in df.columns:
        df["conversation_id"] = ""
    df[text_column] = df[text_column].fillna("").astype(str)
    df[original_text_column] = df[original_text_column].fillna("").astype(str)
    df = df[df[text_column].str.strip().astype(bool)].copy()
    if max_messages is not None:
        df = df.head(max_messages).copy()
    return df


def normalize_entity_text(text: str) -> str:
    text = str(text).replace("##", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\n\r.,;:!?()[]{}<>\"'`´“”‚‘")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text: str) -> str:
    text = normalize_entity_text(text).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-zA-ZäöüÄÖÜ0-9_+\-/ ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def should_exclude_entity(entity_text: str) -> bool:
    key = normalize_key(entity_text)
    if not key:
        return True
    if len(key) == 1 and not key.isdigit():
        return True
    if key in DEFAULT_EXCLUSION_NORMALIZED:
        return True
    # Exclude pure numbers, dates, times and common anonymization-like markers.
    if re.fullmatch(r"\d+([:./-]\d+)*", key):
        return True
    if re.fullmatch(r"person\d+|vorname|nachname", key):
        return True
    return False


def standardize_entity_type(raw_type: str, backend: str) -> str:
    if raw_type in STANDARD_TYPE_MAP:
        return STANDARD_TYPE_MAP[raw_type]
    raw_clean = re.sub(r"^[BI]-", "", str(raw_type))
    if raw_clean in STANDARD_TYPE_MAP:
        return STANDARD_TYPE_MAP[raw_clean]
    # Some token-classification models expose LABEL_1 etc. For medication-only model,
    # label-like positives are mapped to medication_candidate.
    if backend == "medner" and raw_clean.upper() not in {"O", "LABEL_0"}:
        return "medication_candidate"
    return raw_clean.lower() if raw_clean else "unknown"


def get_context(text: str, start: Optional[int], end: Optional[int], context_chars: int) -> str:
    if start is None or end is None:
        return text[: 2 * context_chars + 80]
    try:
        start_i = max(0, int(start) - context_chars)
        end_i = min(len(text), int(end) + context_chars)
        left = "…" if start_i > 0 else ""
        right = "…" if end_i < len(text) else ""
        return left + text[start_i:end_i] + right
    except Exception:
        return text[: 2 * context_chars + 80]


def device_to_pipeline_arg(device: str) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    # auto: choose cuda if available, else cpu
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def load_transformers_pipeline(model_name: str, device: str):
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Install with: pip install transformers torch") from exc

    LOGGER.info("Loading transformers token-classification model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_to_pipeline_arg(device),
    )


def extract_with_transformers(
    df: pd.DataFrame,
    pipe: Any,
    backend: str,
    model_name: str,
    text_column: str,
    original_text_column: str,
    min_score: float,
    context_chars: int,
    include_full_message: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{backend} NER"):
        text = str(row[text_column])
        original = str(row[original_text_column])
        if not text.strip():
            continue
        try:
            preds = pipe(text, truncation=True)
        except TypeError:
            # Some older pipelines do not accept truncation kwarg.
            preds = pipe(text)
        except Exception as exc:
            LOGGER.debug("NER failed for message %s: %s", row.get("id", ""), exc)
            continue

        for pred in preds:
            entity_text = normalize_entity_text(pred.get("word", ""))
            if should_exclude_entity(entity_text):
                continue
            score = float(pred.get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            raw_type = str(pred.get("entity_group") or pred.get("entity") or "")
            start = pred.get("start")
            end = pred.get("end")
            standard_type = standardize_entity_type(raw_type, backend)
            out = {
                "backend": backend,
                "model_name": model_name,
                "corpus_type": row.get("corpus_type", ""),
                "message_id": row.get("id", ""),
                "conversation_id": row.get("conversation_id", ""),
                "entity_text": entity_text,
                "entity_key": normalize_key(entity_text),
                "entity_type_raw": raw_type,
                "entity_type_standard": standard_type,
                "score": score,
                "start": start,
                "end": end,
                "context": get_context(original if original else text, start, end, context_chars),
            }
            if include_full_message:
                out["text_original"] = original
                out["text_clean_lexical"] = text
            rows.append(out)
    return rows


def load_gliner_model(model_name: str):
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError("Missing dependency for GLiNER. Install with: pip install gliner") from exc

    LOGGER.info("Loading GLiNER model: %s", model_name)
    return GLiNER.from_pretrained(model_name)


def extract_with_gliner(
    df: pd.DataFrame,
    model: Any,
    model_name: str,
    labels: List[str],
    text_column: str,
    original_text_column: str,
    threshold: float,
    context_chars: int,
    include_full_message: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="gliner NER"):
        text = str(row[text_column])
        original = str(row[original_text_column])
        if not text.strip():
            continue
        try:
            ents = model.predict_entities(text, labels, threshold=threshold)
        except Exception as exc:
            LOGGER.debug("GLiNER failed for message %s: %s", row.get("id", ""), exc)
            continue
        for ent in ents:
            entity_text = normalize_entity_text(ent.get("text", ""))
            if should_exclude_entity(entity_text):
                continue
            raw_type = str(ent.get("label", ""))
            score = float(ent.get("score", 0.0) or 0.0)
            start = ent.get("start")
            end = ent.get("end")
            out = {
                "backend": "gliner",
                "model_name": model_name,
                "corpus_type": row.get("corpus_type", ""),
                "message_id": row.get("id", ""),
                "conversation_id": row.get("conversation_id", ""),
                "entity_text": entity_text,
                "entity_key": normalize_key(entity_text),
                "entity_type_raw": raw_type,
                "entity_type_standard": standardize_entity_type(raw_type, "gliner"),
                "score": score,
                "start": start,
                "end": end,
                "context": get_context(original if original else text, start, end, context_chars),
            }
            if include_full_message:
                out["text_original"] = original
                out["text_clean_lexical"] = text
            rows.append(out)
    return rows


def choose_backends(backend: str) -> List[str]:
    if backend == "all":
        return ["humadex", "medner", "gliner"]
    if backend == "humadex_medner":
        return ["humadex", "medner"]
    return [backend]


def aggregate_candidates(mentions: pd.DataFrame, min_occurrences: int, min_message_count: int, max_kwic: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mentions.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    group_cols = ["entity_key", "entity_text", "entity_type_standard"]
    summary = (
        mentions.groupby(group_cols, dropna=False)
        .agg(
            total_occurrences=("entity_text", "size"),
            message_count=("message_id", pd.Series.nunique),
            corpus_types=("corpus_type", lambda x: ";".join(sorted(set(map(str, x))))),
            backends=("backend", lambda x: ";".join(sorted(set(map(str, x))))),
            raw_labels=("entity_type_raw", lambda x: ";".join(sorted(set(map(str, x))))),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
        )
        .reset_index()
    )
    summary = summary[
        (summary["total_occurrences"] >= min_occurrences)
        & (summary["message_count"] >= min_message_count)
    ].copy()
    summary = summary.sort_values(["total_occurrences", "message_count", "mean_score"], ascending=[False, False, False])

    # Add review columns for manual validation.
    summary.insert(0, "review_decision", "")
    summary.insert(1, "final_entity_type", "")
    summary.insert(2, "final_preferred_term", "")
    summary.insert(3, "review_note", "")

    by_corpus = (
        mentions.groupby(["entity_key", "entity_text", "entity_type_standard", "corpus_type"], dropna=False)
        .agg(
            occurrences=("entity_text", "size"),
            message_count=("message_id", pd.Series.nunique),
            mean_score=("score", "mean"),
            backends=("backend", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["entity_type_standard", "occurrences"], ascending=[True, False])
    )

    # KWIC contexts for candidates in summary.
    keep_keys = set(summary["entity_key"].astype(str))
    kwic = mentions[mentions["entity_key"].astype(str).isin(keep_keys)].copy()
    kwic = kwic.sort_values(["entity_key", "score"], ascending=[True, False])
    kwic = kwic.groupby("entity_key", group_keys=False).head(max_kwic).reset_index(drop=True)

    return summary, by_corpus, kwic


def write_outputs(
    output_dir: Path,
    mentions: pd.DataFrame,
    candidates: pd.DataFrame,
    by_corpus: pd.DataFrame,
    kwic: pd.DataFrame,
    config: Dict[str, Any],
    write_csv: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "model_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    readme = pd.DataFrame(
        {
            "field": [
                "purpose",
                "interpretation",
                "manual_review",
                "recommended_review_decisions",
                "publication_note",
            ],
            "value": [
                "Open-source model-based extraction of candidate clinical entities from German chat messages.",
                "Model outputs are candidates, not final annotations.",
                "Fill review_decision, final_entity_type, final_preferred_term and review_note in candidates_for_review.",
                "include_diagnosis; include_medication; include_symptom_or_risk; include_procedure; exclude_nonmedical; exclude_organizational; ambiguous; merge_variant",
                "Report as model-supported candidate extraction with manual contextual validation; do not infer patient prevalence.",
            ],
        }
    )

    summary_by_type = (
        mentions.groupby(["entity_type_standard", "backend"], dropna=False)
        .agg(
            occurrences=("entity_text", "size"),
            unique_candidates=("entity_key", pd.Series.nunique),
            messages=("message_id", pd.Series.nunique),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["occurrences"], ascending=False)
        if not mentions.empty
        else pd.DataFrame()
    )

    summary_by_corpus = (
        mentions.groupby(["corpus_type", "entity_type_standard"], dropna=False)
        .agg(
            occurrences=("entity_text", "size"),
            unique_candidates=("entity_key", pd.Series.nunique),
            messages=("message_id", pd.Series.nunique),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["corpus_type", "occurrences"], ascending=[True, False])
        if not mentions.empty
        else pd.DataFrame()
    )

    excel_path = output_dir / "model_entity_candidates_for_review_SJ.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        summary_by_type.to_excel(writer, sheet_name="summary_by_type", index=False)
        summary_by_corpus.to_excel(writer, sheet_name="summary_by_corpus", index=False)
        candidates.to_excel(writer, sheet_name="candidates_for_review", index=False)
        by_corpus.to_excel(writer, sheet_name="candidate_by_corpus", index=False)
        kwic.to_excel(writer, sheet_name="kwic_contexts", index=False)
        mentions.to_excel(writer, sheet_name="all_model_mentions", index=False)

    # Separate workbooks for easier sharing/archiving.
    mentions.to_excel(output_dir / "model_entity_mentions_long.xlsx", index=False)
    summary_by_corpus.to_excel(output_dir / "model_entity_summary_by_corpus.xlsx", index=False)
    summary_by_type.to_excel(output_dir / "model_entity_summary_by_type.xlsx", index=False)

    if write_csv:
        mentions.to_csv(output_dir / "model_entity_mentions_long.csv", index=False)
        candidates.to_csv(output_dir / "model_entity_candidates_for_review.csv", index=False)
        by_corpus.to_csv(output_dir / "model_entity_by_corpus.csv", index=False)
        kwic.to_csv(output_dir / "model_entity_kwic_contexts.csv", index=False)
        summary_by_corpus.to_csv(output_dir / "model_entity_summary_by_corpus.csv", index=False)
        summary_by_type.to_csv(output_dir / "model_entity_summary_by_type.csv", index=False)

    LOGGER.info("Wrote review workbook: %s", excel_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    project_root = Path(args.project_root).resolve()
    output_dir = resolve_path(project_root, args.output_dir)
    specs = get_corpus_specs(args, project_root)

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Output dir: %s", output_dir)

    dfs = [load_corpus(spec, args.text_column, args.original_text_column, args.max_messages) for spec in specs]
    corpus_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    LOGGER.info("Loaded %d messages across corpus=%s", len(corpus_df), args.corpus)

    all_mentions: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    backends = choose_backends(args.backend)

    for backend in backends:
        try:
            if backend == "humadex":
                pipe = load_transformers_pipeline(args.humadex_model, args.device)
                all_mentions.extend(
                    extract_with_transformers(
                        corpus_df,
                        pipe,
                        backend="humadex",
                        model_name=args.humadex_model,
                        text_column=args.text_column,
                        original_text_column=args.original_text_column,
                        min_score=args.min_score,
                        context_chars=args.context_chars,
                        include_full_message=args.include_full_message,
                    )
                )
            elif backend == "medner":
                pipe = load_transformers_pipeline(args.medner_model, args.device)
                all_mentions.extend(
                    extract_with_transformers(
                        corpus_df,
                        pipe,
                        backend="medner",
                        model_name=args.medner_model,
                        text_column=args.text_column,
                        original_text_column=args.original_text_column,
                        min_score=args.min_score,
                        context_chars=args.context_chars,
                        include_full_message=args.include_full_message,
                    )
                )
            elif backend == "gliner":
                labels = [x.strip() for x in args.gliner_labels.split(",") if x.strip()]
                model = load_gliner_model(args.gliner_model)
                all_mentions.extend(
                    extract_with_gliner(
                        corpus_df,
                        model,
                        model_name=args.gliner_model,
                        labels=labels,
                        text_column=args.text_column,
                        original_text_column=args.original_text_column,
                        threshold=args.gliner_threshold,
                        context_chars=args.context_chars,
                        include_full_message=args.include_full_message,
                    )
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as exc:
            LOGGER.error("Backend '%s' failed: %s", backend, exc)
            errors.append({"backend": backend, "error": str(exc)})

    mentions = pd.DataFrame(all_mentions)
    if mentions.empty:
        LOGGER.warning("No model mentions were extracted. Check model installation, thresholds, and text column.")
        mentions = pd.DataFrame(
            columns=[
                "backend",
                "model_name",
                "corpus_type",
                "message_id",
                "conversation_id",
                "entity_text",
                "entity_key",
                "entity_type_raw",
                "entity_type_standard",
                "score",
                "start",
                "end",
                "context",
            ]
        )

    candidates, by_corpus, kwic = aggregate_candidates(
        mentions,
        min_occurrences=args.min_occurrences,
        min_message_count=args.min_message_count,
        max_kwic=args.max_kwic_per_candidate,
    )

    config = vars(args).copy()
    config.update(
        {
            "project_root_resolved": str(project_root),
            "input_files": [str(s.path) for s in specs],
            "n_messages_loaded": int(len(corpus_df)),
            "n_model_mentions": int(len(mentions)),
            "n_candidates_for_review": int(len(candidates)),
            "backend_errors": errors,
        }
    )

    write_outputs(output_dir, mentions, candidates, by_corpus, kwic, config, args.write_csv)
    LOGGER.info("Extracted %d model mentions and %d candidate rows for review.", len(mentions), len(candidates))
    if errors:
        LOGGER.warning("Some backends failed. See model_run_config.json for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
