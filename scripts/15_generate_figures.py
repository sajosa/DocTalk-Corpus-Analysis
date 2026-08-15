#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_generate_thread_length_figure.py

Generate the publication-ready thread-length distribution figure for the
DocTalk corpus.

The script is intentionally single-purpose: it creates Figure 1 only.
It does not generate temporal heatmaps, avoiding overlap with the dedicated
time-analysis script.

Inputs
------
outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical_v2.csv
outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical_v2.csv

Outputs
-------
outputs/public/figures/
outputs/public/tables/figure_sources/thread_size_distribution_table.csv

The final corpus is expected to contain:
- 293 direct-message conversations
- 86 group-message threads
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THREAD_BINS = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    (">100", 101, np.inf),
]

DIRECT_COLOR = "#D9D9D9"
GROUP_COLOR = "#595959"
EDGE_COLOR = "#303030"
GRID_COLOR = "#E0E0E0"
PANEL_BG = "#F7F7F7"


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
        description="Generate the publication-ready thread-length figure for the DocTalk corpus."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--direct-path",
        default="outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical_v2.csv",
    )
    parser.add_argument(
        "--group-path",
        default="outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical_v2.csv",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/public/figures",
    )
    parser.add_argument(
        "--tables-dir",
        default="outputs/public/tables/figure_sources",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--with-internal-titles",
        action="store_true",
        help="Keep titles inside figures. Otherwise use captions in manuscript.",
    )
    parser.add_argument("--no-source-tables", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def thread_bin_label(n: int) -> str:
    for label, lo, hi in THREAD_BINS:
        if lo <= n <= hi:
            return label
    return ">100"


def build_thread_size_distribution(df: pd.DataFrame, corpus_label: str) -> pd.DataFrame:
    thread_counts = (
        df.groupby("conversation_id", dropna=False)
        .size()
        .reset_index(name="n_utterances")
    )
    thread_counts["thread_bin"] = thread_counts["n_utterances"].apply(thread_bin_label)

    categories = [x[0] for x in THREAD_BINS]

    out = (
        thread_counts["thread_bin"]
        .value_counts()
        .reindex(categories, fill_value=0)
        .rename_axis("thread_bin")
        .reset_index(name="n_threads")
    )

    total = out["n_threads"].sum()
    out["percentage"] = out["n_threads"] / total * 100
    out["corpus"] = corpus_label

    return out[["corpus", "thread_bin", "n_threads", "percentage"]]


def plot_thread_size_distribution(
    direct_dist: pd.DataFrame,
    group_dist: pd.DataFrame,
    out_base: Path,
    dpi: int,
    with_internal_titles: bool,
) -> None:
    categories = [x[0] for x in THREAD_BINS]

    direct_plot = direct_dist.set_index("thread_bin").loc[categories].reset_index()
    group_plot = group_dist.set_index("thread_bin").loc[categories].reset_index()

    y = np.arange(len(categories))
    bar_h = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_facecolor(PANEL_BG)

    ax.barh(
        y - bar_h / 2,
        direct_plot["percentage"],
        height=bar_h,
        color=DIRECT_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label="Direct messages",
    )
    ax.barh(
        y + bar_h / 2,
        group_plot["percentage"],
        height=bar_h,
        color=GROUP_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label="Group messages",
    )

    for i, row in direct_plot.iterrows():
        ax.text(
            row["percentage"] + 0.7,
            i - bar_h / 2,
            f'{row["percentage"]:.1f}% (n={int(row["n_threads"])})',
            va="center",
            ha="left",
            fontsize=9,
            color=EDGE_COLOR,
        )

    for i, row in group_plot.iterrows():
        ax.text(
            row["percentage"] + 0.7,
            i + bar_h / 2,
            f'{row["percentage"]:.1f}% (n={int(row["n_threads"])})',
            va="center",
            ha="left",
            fontsize=9,
            color=EDGE_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()

    ax.set_xlabel("Threads (%)")
    ax.set_ylabel("Messages per thread")

    if with_internal_titles:
        ax.set_title("Distribution of thread length by communication modality", pad=12)

    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = max(direct_plot["percentage"].max(), group_plot["percentage"].max())
    ax.set_xlim(0, xmax + 14)

    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_final_thread_counts(
    direct_dist: pd.DataFrame,
    group_dist: pd.DataFrame,
    expected_direct_threads: int = 293,
    expected_group_threads: int = 86,
) -> None:
    """Validate that Figure 1 is based on the locked final corpus structure."""
    direct_total = int(direct_dist["n_threads"].sum())
    group_total = int(group_dist["n_threads"].sum())

    if direct_total != expected_direct_threads:
        raise ValueError(
            f"Unexpected number of direct-message conversations: {direct_total}. "
            f"Expected {expected_direct_threads}."
        )

    if group_total != expected_group_threads:
        raise ValueError(
            f"Unexpected number of group-message threads: {group_total}. "
            f"Expected {expected_group_threads}."
        )

    print(
        "Final thread-count validation passed: "
        f"{direct_total} direct-message conversations and "
        f"{group_total} group-message threads."
    )


def main() -> None:
    args = parse_args()
    set_publication_style()

    project_root = Path(args.project_root).resolve()
    direct_path = resolve_path(project_root, args.direct_path)
    group_path = resolve_path(project_root, args.group_path)
    figures_dir = resolve_path(project_root, args.figures_dir)
    tables_dir = resolve_path(project_root, args.tables_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_source_tables:
        tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Direct input: {direct_path}")
    print(f"Group input:  {group_path}")
    print(f"Figures dir:  {figures_dir}")

    direct_df = load_csv(direct_path)
    group_df = load_csv(group_path)

    required_column = "conversation_id"
    for label, df in [("direct", direct_df), ("group", group_df)]:
        if required_column not in df.columns:
            raise ValueError(
                f"Required column '{required_column}' missing from {label} input. "
                f"Available columns: {df.columns.tolist()}"
            )

    print(f"Loaded direct: {direct_df.shape}")
    print(f"Loaded group:  {group_df.shape}")

    direct_dist = build_thread_size_distribution(
        direct_df,
        "direct",
    )
    group_dist = build_thread_size_distribution(
        group_df,
        "group",
    )

    validate_final_thread_counts(
        direct_dist,
        group_dist,
    )

    if not args.no_source_tables:
        source_path = (
            tables_dir
            / "thread_size_distribution_table.csv"
        )
        pd.concat(
            [direct_dist, group_dist],
            ignore_index=True,
        ).to_csv(
            source_path,
            index=False,
            encoding="utf-8",
        )
        print("Saved figure-source table:")
        print(source_path)

    out_base = (
        figures_dir
        / "combined_direct_group_thread_length_distribution"
    )

    plot_thread_size_distribution(
        direct_dist=direct_dist,
        group_dist=group_dist,
        out_base=out_base,
        dpi=args.dpi,
        with_internal_titles=args.with_internal_titles,
    )

    print("Created Figure 1:")
    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".svg"))
    print(out_base.with_suffix(".pdf"))
    print("Done.")


if __name__ == "__main__":
    main()