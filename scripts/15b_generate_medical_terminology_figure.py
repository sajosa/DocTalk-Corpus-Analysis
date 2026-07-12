#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15b_generate_medical_terminology_figure.py

Generate a publication-ready figure for the exploratory clinical terminology
analysis.

Figure
------
Validated clinical entity mentions per 1,000 cleaned tokens by entity type
and message type.

Input
-----
outputs/public/tables/medical_terminology/
validated_clinical_terminology_summary.xlsx

Required sheet:
by_type_comparison

Outputs
-------
outputs/public/figures/medical_terminology/
figure_clinical_terminology_density_by_entity_type.png
figure_clinical_terminology_density_by_entity_type.svg
figure_clinical_terminology_density_by_entity_type.pdf

outputs/public/tables/figure_sources/
medical_terminology_density_by_entity_type_table.csv

Interpretation
--------------
Communication-level terminology density, not patient-level prevalence.

Usage
-----
No title:
    python scripts/15b_generate_medical_terminology_figure.py

With internal title:
    python scripts/15b_generate_medical_terminology_figure.py --with-internal-titles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIRECT_COLOR = "#D9D9D9"
GROUP_COLOR = "#595959"
EDGE_COLOR = "#303030"
GRID_COLOR = "#E0E0E0"
PANEL_BG = "#F7F7F7"

ENTITY_LABELS = {
    "symptom": "Symptom",
    "medication": "Medication",
    "diagnosis": "Diagnosis",
    "procedure": "Procedure",
    "therapy": "Therapy",
    "treatment": "Treatment",
    "clinical_risk": "Clinical risk",
}

ENTITY_ORDER = [
    "symptom",
    "medication",
    "diagnosis",
    "procedure",
    "therapy",
    "treatment",
    "clinical_risk",
]


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready clinical terminology density figure."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--input-file",
        default=(
            "outputs/public/tables/medical_terminology/"
            "validated_clinical_terminology_summary.xlsx"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/public/figures/medical_terminology",
    )
    parser.add_argument(
        "--tables-dir",
        default="outputs/public/tables/figure_sources",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--with-internal-titles",
        action="store_true",
        help="Keep title inside figure. Otherwise use manuscript caption.",
    )
    parser.add_argument("--no-source-tables", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_terminology_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path, sheet_name="by_type_comparison")

    required_columns = [
        "final_entity_type",
        "direct_mentions_per_1000_tokens",
        "group_mentions_per_1000_tokens",
        "direct_mentions",
        "group_mentions",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in by_type_comparison: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["final_entity_type"] = df["final_entity_type"].astype(str).str.strip()

    unexpected = sorted(set(df["final_entity_type"]) - set(ENTITY_ORDER))
    if unexpected:
        print(f"Warning: unexpected entity types ignored: {unexpected}")

    df = df[df["final_entity_type"].isin(ENTITY_ORDER)].copy()

    if df.empty:
        raise ValueError("No expected entity types found.")

    order_map = {entity_type: i for i, entity_type in enumerate(ENTITY_ORDER)}
    df["entity_order"] = df["final_entity_type"].map(order_map)
    df["entity_label"] = df["final_entity_type"].map(ENTITY_LABELS)

    df = df.sort_values("entity_order").reset_index(drop=True)

    return df


def plot_clinical_terminology_density(
    df: pd.DataFrame,
    out_base: Path,
    dpi: int,
    with_internal_titles: bool,
) -> None:
    labels = df["entity_label"].tolist()

    direct_rates = df["direct_mentions_per_1000_tokens"].astype(float).to_numpy()
    group_rates = df["group_mentions_per_1000_tokens"].astype(float).to_numpy()

    direct_counts = df["direct_mentions"].astype(int).to_numpy()
    group_counts = df["group_mentions"].astype(int).to_numpy()

    y = np.arange(len(labels))
    bar_h = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_facecolor(PANEL_BG)

    ax.barh(
        y - bar_h / 2,
        direct_rates,
        height=bar_h,
        color=DIRECT_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label="Direct messages",
    )

    ax.barh(
        y + bar_h / 2,
        group_rates,
        height=bar_h,
        color=GROUP_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label="Group messages",
    )

    max_rate = max(float(direct_rates.max()), float(group_rates.max()))

    for i, (direct_rate, direct_n) in enumerate(zip(direct_rates, direct_counts)):
        ax.text(
            direct_rate + max_rate * 0.025,
            i - bar_h / 2,
            f"{direct_rate:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color=EDGE_COLOR,
        )

    for i, (group_rate, group_n) in enumerate(zip(group_rates, group_counts)):
        ax.text(
            group_rate + max_rate * 0.025,
            i + bar_h / 2,
            f"{group_rate:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color=EDGE_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Mentions per 1,000 cleaned tokens")
    ax.set_ylabel("Entity type")

    if with_internal_titles:
        ax.set_title(
            "Clinical terminology density by message type",
            pad=12,
        )

    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0, max_rate + 1.8)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_publication_style()

    project_root = Path(args.project_root).resolve()
    input_file = resolve_path(project_root, args.input_file)
    figures_dir = resolve_path(project_root, args.figures_dir)
    tables_dir = resolve_path(project_root, args.tables_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_source_tables:
        tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Input file:   {input_file}")
    print(f"Figures dir:  {figures_dir}")

    df = load_terminology_table(input_file)

    if not args.no_source_tables:
        source_table = df[
            [
                "final_entity_type",
                "entity_label",
                "direct_mentions_per_1000_tokens",
                "group_mentions_per_1000_tokens",
                "direct_mentions",
                "group_mentions",
                "direct_vs_group_rate_ratio",
            ]
        ].copy()

        source_table.to_csv(
            tables_dir / "medical_terminology_density_by_entity_type_table.csv",
            index=False,
        )

    plot_clinical_terminology_density(
        df=df,
        out_base=figures_dir / "figure_clinical_terminology_density_by_entity_type",
        dpi=args.dpi,
        with_internal_titles=args.with_internal_titles,
    )

    print("Done.")


if __name__ == "__main__":
    main()