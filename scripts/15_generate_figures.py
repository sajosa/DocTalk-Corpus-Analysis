#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate publication-ready figures for the DocTalk corpus analysis.

Figures
-------
1) Utterance-count distribution by message type
2) Weekday-hour heatmap by message type

Inputs
------
outputs/confidential/dataframes/D_utterances_raw.csv
outputs/confidential/dataframes/G_utterances_raw.csv

Outputs
-------
outputs/public/figures/
outputs/public/tables/figure_sources/

workflow
-------
no title : python scripts/16_generate_figures.py 2>&1 | tee audit/logs/16_generate_figures.log
with title : python scripts/16_generate_figures.py --with-internal-titles 2>&1 | tee audit/logs/16_generate_figures.log

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

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
        description="Generate publication-ready figures for the DocTalk corpus analysis."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--direct-path",
        default="outputs/confidential/dataframes/D_utterances_raw.csv",
    )
    parser.add_argument(
        "--group-path",
        default="outputs/confidential/dataframes/G_utterances_raw.csv",
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
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=4.0,
        help="Fixed maximum for heatmap color scale. Default: 4.0.",
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


def ensure_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Expected column 'timestamp' not found.")

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp_parsed"] = pd.to_datetime(
            df["timestamp"], unit="s", errors="coerce"
        )
    else:
        df["timestamp_parsed"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

    df["hour"] = df["timestamp_parsed"].dt.hour

    if "weekday" in df.columns:
        df["weekday_clean"] = df["weekday"].astype(str).str.strip()
    else:
        df["weekday_clean"] = df["timestamp_parsed"].dt.day_name()

    return df


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

    ax.set_xlabel("Percentage of message threads")
    ax.set_ylabel("Number of utterances per thread")

    if with_internal_titles:
        ax.set_title("Utterance-count distribution by message type", pad=12)

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


def build_weekday_hour_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime_columns(df)
    tmp = df.dropna(subset=["hour"]).copy()
    tmp["hour"] = tmp["hour"].astype(int)

    matrix = pd.crosstab(tmp["weekday_clean"], tmp["hour"])
    matrix = matrix.reindex(index=WEEKDAY_ORDER, fill_value=0)
    matrix = matrix.reindex(columns=list(range(24)), fill_value=0)

    total = matrix.to_numpy().sum()
    if total == 0:
        return matrix.astype(float)

    return matrix / total * 100


def plot_weekday_hour_heatmaps(
    direct_pct: pd.DataFrame,
    group_pct: pd.DataFrame,
    out_base: Path,
    dpi: int,
    heatmap_vmax: float,
) -> None:
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[32, 1.1],
        height_ratios=[1, 1],
        wspace=0.10,
        hspace=0.23,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])

    im1 = ax1.imshow(
        direct_pct.values,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=heatmap_vmax,
        interpolation="nearest",
    )

    im2 = ax2.imshow(
        group_pct.values,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=heatmap_vmax,
        interpolation="nearest",
    )

    xticks = np.arange(0, 24, 2)
    xticklabels = [f"{h:02d}:00" for h in xticks]

    for ax in [ax1, ax2]:
        ax.set_yticks(np.arange(len(WEEKDAY_ORDER)))
        ax.set_yticklabels(WEEKDAY_ORDER)
        ax.set_ylabel("Weekday")
        ax.set_xticks(xticks)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_xticklabels([])
    ax2.set_xticklabels(xticklabels)
    ax2.set_xlabel("Hour of day")

    ax1.set_title("A. Direct messages", loc="left", pad=8)
    ax2.set_title("B. Group messages", loc="left", pad=8)

    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("Messages within each corpus (%)")
    cb.ax.tick_params(labelsize=9.5)

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
    direct_path = resolve_path(project_root, args.direct_path)
    group_path = resolve_path(project_root, args.group_path)
    figures_dir = resolve_path(project_root, args.figures_dir)
    tables_dir = resolve_path(project_root, args.tables_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "time").mkdir(parents=True, exist_ok=True)

    if not args.no_source_tables:
        tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Direct input: {direct_path}")
    print(f"Group input:  {group_path}")
    print(f"Figures dir:  {figures_dir}")

    direct_df = load_csv(direct_path)
    group_df = load_csv(group_path)

    print(f"Loaded direct: {direct_df.shape}")
    print(f"Loaded group:  {group_df.shape}")

    direct_dist = build_thread_size_distribution(direct_df, "direct")
    group_dist = build_thread_size_distribution(group_df, "group")

    if not args.no_source_tables:
        pd.concat([direct_dist, group_dist], ignore_index=True).to_csv(
            tables_dir / "thread_size_distribution_table.csv",
            index=False,
        )

    plot_thread_size_distribution(
        direct_dist=direct_dist,
        group_dist=group_dist,
        out_base=figures_dir / "combined_direct_group_utterance_count_categories_horizontal_bar",
        dpi=args.dpi,
        with_internal_titles=args.with_internal_titles,
    )

    direct_pct = build_weekday_hour_matrix(direct_df)
    group_pct = build_weekday_hour_matrix(group_df)

    if not args.no_source_tables:
        direct_pct.to_csv(tables_dir / "direct_weekday_hour_percentage_table.csv")
        group_pct.to_csv(tables_dir / "group_weekday_hour_percentage_table.csv")

    plot_weekday_hour_heatmaps(
        direct_pct=direct_pct,
        group_pct=group_pct,
        out_base=figures_dir / "time" / "combined_direct_group_weekday_hour_heatmap",
        dpi=args.dpi,
        heatmap_vmax=args.heatmap_vmax,
    )

    print("Done.")


if __name__ == "__main__":
    main()