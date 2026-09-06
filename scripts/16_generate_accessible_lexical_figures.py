#!/usr/bin/env python3
"""
16_generate_accessible_lexical_figures.py

Generate two accessible, publication-ready figures for the DocTalk corpus:

1. A two-panel dumbbell plot of the highest-ranking token-keyness items.
   Token selection, keyness statistics, and normalized direct/group
   frequencies are taken from the final keyness review workbook so that the
   displayed frequencies use exactly the same token universe as the keyness
   analysis.

2. A low-minus-high volume difference plot for the same top-keyness tokens.
   This supplementary contrast is calculated from the final cleaned lexical
   v2 corpus tables.

The script writes only aggregated public outputs. It does not export message
text or KWIC context.

Default inputs:
    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical_v2.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical_v2.csv

Default outputs:
    outputs/public/figures/accessible_results/
    outputs/public/tables/figure_sources/

Peak/low-volume definition:
    For each corpus separately, message volume is counted for every occupied
    weekday-hour cell. Cells at or above the 75th percentile are classified as
    high-volume; cells at or below the 25th percentile are classified as
    low-volume. Middle-volume cells are excluded from the contrast.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIRECT_COLOR = "#D9D9D9"
GROUP_COLOR = "#595959"
EDGE_COLOR = "#303030"
GRID_COLOR = "#E0E0E0"
PANEL_BG = "#F7F7F7"
WHITE = "#FFFFFF"

DIRECTION_LABELS = {
    "direct": "Direct messages",
    "group": "Group messages",
}

DEFAULT_MARKERS = [
    "PatName",
    "Hashtag_PatName",
    "KolName",
    "Mention_KolName",
    "Übergabe",
    "Rückmeldung",
    "WE",
    "Todo",
    "kein_Todo",
    "Raum",
    "ÖGD",
    "MT_Gruppe",
    "GT_Gruppe",
    "KT_Gruppe",
]

DISPLAY_LABELS = {
    "PatName": "Patient reference",
    "Hashtag_PatName": "Hashtagged patient reference",
    "KolName": "Colleague reference",
    "Mention_KolName": "Direct colleague mention",
    "Übergabe": "Handover",
    "Rückmeldung": "Feedback / status update",
    "WE": "Weekend",
    "Todo": "Task / to-do",
    "kein_Todo": "No active to-do",
    "Raum": "Room",
    "ÖGD": "Upper GI endoscopy",
    "MT_Gruppe": "Music therapy group",
    "GT_Gruppe": "Talk therapy group",
    "KT_Gruppe": "Art therapy group",
}


TOP_KEYNESS_DISPLAY_LABELS = {
    "du": "Second-person pronoun (du)",
    "ich": "First-person pronoun (ich)",
    "QuestionMark": "Question marker (?)",
    "dir": "Second-person pronoun (dir)",
    "ja": "Affirmation (ja)",
    "hast": "Verb form (hast)",
    "Hi": "Informal greeting (Hi)",
    "ok": "Acknowledgement (ok)",
    "PatName": "Patient reference (PatName)",
    "Hashtag_PatName": "Hashtagged patient reference",
    "kein_Todo": "No active to-do",
    "Mention_KolName": "Colleague mention",
    "anwesend": "Attendance (anwesend)",
    "Gruppe": "Group (Gruppe)",
    "sich": "Reflexive pronoun (sich)",
    "Datum": "Date (Datum)",
    "Rückmeldung": "Feedback / status",
}


TOKEN_PATTERN = re.compile(r"[\wÄÖÜäöüß]+(?:_[\wÄÖÜäöüß]+)*", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate accessible lexical figures comparing direct/group "
            "messages and high-/low-volume time cells."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--direct-path",
        type=Path,
        default=Path(
            "outputs/confidential/cleaned_corpus_tables/"
            "D_utterances_clean_lexical_v2.csv"
        ),
    )
    parser.add_argument(
        "--group-path",
        type=Path,
        default=Path(
            "outputs/confidential/cleaned_corpus_tables/"
            "G_utterances_clean_lexical_v2.csv"
        ),
    )
    parser.add_argument("--text-column", default="text_clean_lexical_v2")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--markers", nargs="+", default=DEFAULT_MARKERS)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument(
        "--keyness-path",
        type=Path,
        default=Path("outputs/public/tables/keyness_review_top_items.xlsx"),
        help="Workbook containing tokens_direct and tokens_group sheets.",
    )
    parser.add_argument(
        "--top-n-per-direction",
        type=int,
        default=8,
        help="Number of highest-ranking nonredundant tokens per direction.",
    )
    parser.add_argument(
        "--min-total-count",
        type=int,
        default=50,
        help="Minimum total token count for automatic keyness selection.",
    )
    parser.add_argument(
        "--keep-case-duplicates",
        action="store_true",
        help="Keep case-only duplicates such as du and Du.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/public/figures/accessible_results"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/public/tables/figure_sources"),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--with-internal-titles", action="store_true")
    parser.add_argument("--case-insensitive", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def validate_quantiles(low: float, high: float) -> None:
    if not 0 <= low < high <= 1:
        raise ValueError(
            "Quantiles must satisfy 0 <= low_quantile < high_quantile <= 1."
        )


def tokenize(text: object, case_insensitive: bool = False) -> list[str]:
    tokens = TOKEN_PATTERN.findall("" if pd.isna(text) else str(text))
    return [token.casefold() for token in tokens] if case_insensitive else tokens



def load_top_keyness_tokens(
    path: Path,
    top_n_per_direction: int,
    min_total_count: int,
    keep_case_duplicates: bool,
) -> tuple[list[str], pd.DataFrame]:
    """Select the highest-ranking token items from the keyness workbook."""
    if not path.exists():
        raise FileNotFoundError(f"Keyness workbook not found: {path}")

    selected_frames: list[pd.DataFrame] = []

    for sheet_name, direction in [
        ("tokens_direct", "direct"),
        ("tokens_group", "group"),
    ]:
        df = pd.read_excel(path, sheet_name=sheet_name)

        required = {
            "token",
            "direction",
            "total_count",
            "direct_freq_per_1000",
            "group_freq_per_1000",
            "log_ratio_direct_vs_group",
            "log_likelihood",
            "signed_log_likelihood",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing keyness columns in sheet '{sheet_name}': "
                f"{sorted(missing)}"
            )

        ranked = df.copy()
        ranked["token"] = ranked["token"].astype(str)
        ranked["total_count"] = pd.to_numeric(
            ranked["total_count"], errors="coerce"
        )
        ranked["log_likelihood"] = pd.to_numeric(
            ranked["log_likelihood"], errors="coerce"
        )
        ranked = ranked[
            (ranked["direction"] == direction)
            & (ranked["total_count"] >= min_total_count)
        ].copy()
        ranked = ranked.sort_values(
            ["log_likelihood", "total_count"],
            ascending=[False, False],
        )

        if not keep_case_duplicates:
            ranked["_dedup_key"] = ranked["token"].str.casefold()
            ranked = ranked.drop_duplicates("_dedup_key", keep="first")

        ranked = ranked.head(top_n_per_direction).copy()
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
        ranked["selected_direction"] = direction
        ranked["display_label"] = ranked["token"].map(
            TOP_KEYNESS_DISPLAY_LABELS
        ).fillna(ranked["token"])
        selected_frames.append(ranked)

    selection = pd.concat(selected_frames, ignore_index=True)
    tokens = selection["token"].drop_duplicates().tolist()
    return tokens, selection


def build_keyness_figure_source(
    selection_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the public source table for the direct-vs-group keyness dumbbell plot.

    All counts, normalized frequencies, and keyness statistics are taken
    directly from the final keyness workbook. This avoids recalculating
    frequencies from a potentially different lexical text representation.
    """
    required = {
        "token",
        "selected_direction",
        "selection_rank",
        "display_label",
        "direct_count",
        "group_count",
        "total_count",
        "direct_freq_per_1000",
        "group_freq_per_1000",
        "log_ratio_direct_vs_group",
        "log_likelihood",
        "signed_log_likelihood",
    }
    missing = required.difference(selection_table.columns)
    if missing:
        raise ValueError(
            "Cannot build keyness figure source table. Missing columns: "
            f"{sorted(missing)}"
        )

    source = selection_table[
        [
            "token",
            "selected_direction",
            "selection_rank",
            "display_label",
            "direct_count",
            "group_count",
            "total_count",
            "direct_freq_per_1000",
            "group_freq_per_1000",
            "log_ratio_direct_vs_group",
            "log_likelihood",
            "signed_log_likelihood",
        ]
    ].copy()

    numeric_columns = [
        "direct_count",
        "group_count",
        "total_count",
        "direct_freq_per_1000",
        "group_freq_per_1000",
        "log_ratio_direct_vs_group",
        "log_likelihood",
        "signed_log_likelihood",
    ]
    for column in numeric_columns:
        source[column] = pd.to_numeric(
            source[column],
            errors="coerce",
        )

    if source[
        [
            "direct_freq_per_1000",
            "group_freq_per_1000",
            "log_likelihood",
        ]
    ].isna().any().any():
        raise ValueError(
            "The keyness figure source contains missing normalized frequencies "
            "or log-likelihood values."
        )

    return source.sort_values(
        ["selected_direction", "selection_rank"],
        ascending=[True, True],
    ).reset_index(drop=True)


def load_corpus(
    path: Path,
    direction: str,
    text_column: str,
    timestamp_column: str,
    markers: Iterable[str],
    case_insensitive: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    if timestamp_column not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' missing in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if text_column not in df.columns:
        fallback_candidates = [
            "text_clean_lexical_v2",
            "interaction_text_v2",
            "text_clean_lexical",
            "text",
        ]
        fallback = next(
            (candidate for candidate in fallback_candidates if candidate in df.columns),
            None,
        )
        if fallback is None:
            raise ValueError(
                f"Text column '{text_column}' missing in {path}, and no supported "
                f"fallback column was found. Available columns: {df.columns.tolist()}"
            )
        print(
            f"Requested text column '{text_column}' not found in {path.name}; "
            f"using '{fallback}' instead."
        )
        text_column = fallback

    out = pd.DataFrame(index=df.index)
    out["direction"] = direction
    out["text"] = df[text_column].fillna("").astype(str)
    out["timestamp"] = pd.to_numeric(df[timestamp_column], errors="coerce")
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="s", errors="coerce")
    out = out.dropna(subset=["datetime"]).copy()
    out["weekday"] = out["datetime"].dt.dayofweek.astype(int)
    out["hour"] = out["datetime"].dt.hour.astype(int)

    marker_keys = [m.casefold() if case_insensitive else m for m in markers]
    token_lists = out["text"].map(
        lambda value: tokenize(value, case_insensitive=case_insensitive)
    )
    out["token_count"] = token_lists.map(len)

    for original_marker, marker_key in zip(markers, marker_keys):
        out[f"marker__{original_marker}"] = token_lists.map(
            lambda tokens, key=marker_key: tokens.count(key)
        )

    if out.empty:
        raise ValueError(f"No valid timestamped rows loaded from {path}.")

    print(
        f"Loaded {len(out):,} {direction} messages; "
        f"{int(out['token_count'].sum()):,} cleaned tokens."
    )
    return out.reset_index(drop=True)


def safe_rate(count: int | float, denominator: int | float) -> float:
    return 0.0 if denominator <= 0 else float(count) / float(denominator) * 1000.0


def classify_volume_cells(
    corpus: pd.DataFrame,
    direction: str,
    low_quantile: float,
    high_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    occupied = (
        corpus.groupby(["weekday", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "message_count"})
    )
    if occupied.empty:
        raise ValueError(f"No occupied time cells for {direction}.")

    low_cutoff = float(occupied["message_count"].quantile(low_quantile))
    high_cutoff = float(occupied["message_count"].quantile(high_quantile))
    occupied["volume_class"] = "middle"
    occupied.loc[occupied["message_count"] <= low_cutoff, "volume_class"] = "low"
    occupied.loc[occupied["message_count"] >= high_cutoff, "volume_class"] = "high"

    if low_cutoff >= high_cutoff:
        ranked = occupied.sort_values(
            ["message_count", "weekday", "hour"]
        ).reset_index(drop=True)
        n_cells = len(ranked)
        n_tail = max(1, int(np.ceil(n_cells * low_quantile)))
        ranked["volume_class"] = "middle"
        ranked.loc[: n_tail - 1, "volume_class"] = "low"
        ranked.loc[n_cells - n_tail :, "volume_class"] = "high"
        occupied = ranked
        low_cutoff = float(
            occupied.loc[occupied["volume_class"] == "low", "message_count"].max()
        )
        high_cutoff = float(
            occupied.loc[occupied["volume_class"] == "high", "message_count"].min()
        )

    classified = corpus.merge(
        occupied[["weekday", "hour", "message_count", "volume_class"]],
        on=["weekday", "hour"],
        how="left",
        validate="many_to_one",
    )

    metadata = {
        "direction": direction,
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "low_cutoff_message_count": low_cutoff,
        "high_cutoff_message_count": high_cutoff,
        "occupied_cells": int(len(occupied)),
        "low_cells": int((occupied["volume_class"] == "low").sum()),
        "high_cells": int((occupied["volume_class"] == "high").sum()),
        "low_messages": int(classified["volume_class"].eq("low").sum()),
        "high_messages": int(classified["volume_class"].eq("high").sum()),
    }
    occupied.insert(0, "direction", direction)
    return classified, occupied, metadata


def build_volume_marker_table(
    classified_frames: list[pd.DataFrame],
    markers: list[str],
) -> pd.DataFrame:
    combined = pd.concat(classified_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for marker in markers:
        for direction in ["direct", "group"]:
            for volume_class in ["low", "high"]:
                sub = combined[
                    (combined["direction"] == direction)
                    & (combined["volume_class"] == volume_class)
                ]
                count = int(sub[f"marker__{marker}"].sum())
                tokens = int(sub["token_count"].sum())
                rows.append(
                    {
                        "marker": marker,
                        "display_label": DISPLAY_LABELS.get(marker, marker),
                        "direction": direction,
                        "message_type": DIRECTION_LABELS[direction],
                        "volume_class": volume_class,
                        "marker_count": count,
                        "messages": int(len(sub)),
                        "messages_with_marker": int(
                            (sub[f"marker__{marker}"] > 0).sum()
                        ),
                        "cleaned_tokens": tokens,
                        "rate_per_1000_tokens": safe_rate(count, tokens),
                    }
                )
    return pd.DataFrame(rows)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": WHITE,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": EDGE_COLOR,
            "xtick.color": EDGE_COLOR,
            "ytick.color": EDGE_COLOR,
            "text.color": EDGE_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "facecolor": WHITE}
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, **kwargs)
    fig.savefig(out_base.with_suffix(".svg"), **kwargs)
    fig.savefig(out_base.with_suffix(".pdf"), **kwargs)
    plt.close(fig)


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.7, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_top_keyness_dumbbell(
    keyness_source: pd.DataFrame,
    out_base: Path,
    dpi: int,
    with_internal_titles: bool,
) -> None:
    """
    Plot normalized frequencies of top keyness-ranked tokens.

    Token selection and ranking are based on log-likelihood from the final
    keyness analysis. The plotted direct/group frequencies are taken directly
    from that same keyness output rather than being recalculated from the
    corpus. Both panels use the same x-axis scale.

    Important:
    The x-axis shows normalized frequency, not a keyness effect size.
    """
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(11.8, 6.8),
        sharex=True,
        constrained_layout=False,
    )

    panel_data: list[pd.DataFrame] = []
    overall_max = 0.0

    for direction in ["direct", "group"]:
        plot_df = (
            keyness_source[
                keyness_source["selected_direction"] == direction
            ]
            .sort_values("selection_rank", ascending=False)
            .copy()
        )

        if plot_df.empty:
            raise ValueError(
                f"No selected keyness items available for direction '{direction}'."
            )

        panel_data.append(plot_df)

        overall_max = max(
            overall_max,
            float(plot_df["direct_freq_per_1000"].max()),
            float(plot_df["group_freq_per_1000"].max()),
        )

    x_limit = max(
        10.0,
        np.ceil(overall_max / 5.0) * 5.0,
    )

    for ax, plot_df, direction, panel in zip(
        axes,
        panel_data,
        ["direct", "group"],
        ["A", "B"],
    ):
        y = np.arange(len(plot_df))

        direct = (
            plot_df["direct_freq_per_1000"]
            .to_numpy(dtype=float)
        )
        group = (
            plot_df["group_freq_per_1000"]
            .to_numpy(dtype=float)
        )

        ax.set_facecolor(PANEL_BG)

        for idx, (direct_value, group_value) in enumerate(
            zip(direct, group)
        ):
            ax.plot(
                [direct_value, group_value],
                [idx, idx],
                color=GRID_COLOR,
                linewidth=2.0,
                zorder=1,
            )

        ax.scatter(
            direct,
            y,
            s=64,
            color=DIRECT_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            label="Direct messages",
            zorder=3,
        )

        ax.scatter(
            group,
            y,
            s=64,
            color=GROUP_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            label="Group messages",
            zorder=3,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(
            plot_df["display_label"]
        )
        ax.set_xlim(0, x_limit)
        ax.set_xlabel(
            "Occurrences per 1,000 tokens"
        )

        panel_title = (
            "Direct-associated tokens"
            if direction == "direct"
            else "Group-associated tokens"
        )

        ax.set_title(
            f"{panel}. {panel_title}",
            loc="center",
            fontweight="bold",
            pad=12,
        )

        style_axis(
            ax,
            grid_axis="x",
        )

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    if with_internal_titles:
        fig.suptitle(
            "Normalized frequencies of top keyness-ranked tokens",
            x=0.5,
            y=0.965,
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    handles, legend_labels = (
        axes[1].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=2.2,
        handletextpad=0.7,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.19,
        top=0.87 if with_internal_titles else 0.93,
        wspace=0.42,
    )

    save_figure(
        fig,
        out_base,
        dpi,
    )


def build_heatmap_matrix(
    volume_table: pd.DataFrame,
    marker_order: list[str],
) -> pd.DataFrame:
    column_order = [
        ("direct", "low"),
        ("direct", "high"),
        ("group", "low"),
        ("group", "high"),
    ]
    pivot = volume_table.pivot_table(
        index="marker",
        columns=["direction", "volume_class"],
        values="rate_per_1000_tokens",
        aggfunc="first",
        fill_value=0.0,
    )
    pivot = pivot.reindex(index=marker_order, fill_value=0.0)
    return pivot.reindex(
        columns=pd.MultiIndex.from_tuples(column_order),
        fill_value=0.0,
    )


def plot_volume_difference(
    volume_table: pd.DataFrame,
    marker_order: list[str],
    out_base: Path,
    dpi: int,
    with_internal_titles: bool,
) -> None:
    """
    Plot low-minus-high volume differences for direct and group messages.

    Positive values indicate higher normalized marker frequency in low-volume
    weekday-hour cells. Negative values indicate relatively higher frequency in
    high-volume cells.
    """
    pivot = volume_table.pivot_table(
        index="marker",
        columns=["direction", "volume_class"],
        values="rate_per_1000_tokens",
        aggfunc="first",
        fill_value=0.0,
    )
    pivot = pivot.reindex(marker_order, fill_value=0.0)

    direct_diff = (
        pivot[("direct", "low")] - pivot[("direct", "high")]
        if ("direct", "high") in pivot.columns
        and ("direct", "low") in pivot.columns
        else pd.Series(0.0, index=pivot.index)
    )
    group_diff = (
        pivot[("group", "low")] - pivot[("group", "high")]
        if ("group", "high") in pivot.columns
        and ("group", "low") in pivot.columns
        else pd.Series(0.0, index=pivot.index)
    )

    plot_df = pd.DataFrame(
        {
            "marker": marker_order,
            "display_label": [
                TOP_KEYNESS_DISPLAY_LABELS.get(marker, marker)
                for marker in marker_order
            ],
            "direct_difference": direct_diff.reindex(marker_order).to_numpy(),
            "group_difference": group_diff.reindex(marker_order).to_numpy(),
        }
    )

    y = np.arange(len(plot_df))
    all_values = np.concatenate(
        [
            plot_df["direct_difference"].to_numpy(),
            plot_df["group_difference"].to_numpy(),
        ]
    )
    max_abs = float(np.nanmax(np.abs(all_values)))
    max_abs = max(1.0, np.ceil(max_abs / 2.0) * 2.0)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(13.4, 7.4),
        sharey=True,
        sharex=True,
        constrained_layout=False,
    )

    for ax, column, direction, panel in [
        (axes[0], "direct_difference", "direct", "A"),
        (axes[1], "group_difference", "group", "B"),
    ]:
        values = plot_df[column].to_numpy()
        ax.set_facecolor(PANEL_BG)
        ax.axvline(0, color=EDGE_COLOR, linewidth=1.0, zorder=2)

        point_colors = [
            GROUP_COLOR if value < 0 else DIRECT_COLOR
            for value in values
        ]

        ax.scatter(
            values,
            y,
            s=68,
            color=point_colors,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            zorder=3,
        )

        for idx, value in enumerate(values):
            ax.plot(
                [0, value],
                [idx, idx],
                color=GRID_COLOR,
                linewidth=2.0,
                zorder=1,
            )

            if value > 0:
                x_text = value + 0.45
                horizontal_alignment = "left"
            elif value < 0:
                x_text = value - 0.45
                horizontal_alignment = "right"
            else:
                x_text = value + 0.45
                horizontal_alignment = "left"

            ax.text(
                x_text,
                idx,
                f"{value:+.1f}",
                ha=horizontal_alignment,
                va="center",
                fontsize=7.6,
                color=EDGE_COLOR,
                clip_on=False,
            )

        ax.set_xlim(-max_abs - 1.8, max_abs + 1.8)
        ax.set_xlabel(
            "Difference per 1,000 tokens\n"
            "(low volume − high volume)",
            labelpad=12,
        )
        ax.set_title(
            f"{panel}. {DIRECTION_LABELS[direction]}",
            loc="center",
            fontweight="bold",
            pad=12,
        )
        style_axis(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["display_label"])
    axes[0].tick_params(axis="y", pad=6)
    axes[1].tick_params(labelleft=False)

    if with_internal_titles:
        fig.suptitle(
            "Change in top keyness-ranked token frequencies by communication volume",
            x=0.5,
            y=0.965,
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    fig.text(
        0.5,
        0.045,
        "Positive values indicate higher relative frequency in low-volume time cells; "
        "negative values indicate higher relative frequency in high-volume time cells.",
        ha="center",
        va="center",
        fontsize=9,
        color=EDGE_COLOR,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.975,
        bottom=0.24,
        top=0.86 if with_internal_titles else 0.93,
        wspace=0.20,
    )

    save_figure(fig, out_base, dpi)


def main() -> None:
    args = parse_args()
    validate_quantiles(args.low_quantile, args.high_quantile)
    set_publication_style()

    root = args.project_root.resolve()
    direct_path = resolve_path(root, args.direct_path)
    group_path = resolve_path(root, args.group_path)
    figures_dir = resolve_path(root, args.figures_dir)
    tables_dir = resolve_path(root, args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    keyness_path = resolve_path(root, args.keyness_path)
    selected_markers, selection_table = load_top_keyness_tokens(
        keyness_path,
        args.top_n_per_direction,
        args.min_total_count,
        args.keep_case_duplicates,
    )
    print("\nAutomatically selected top-keyness tokens:")
    print(
        selection_table[
            [
                "selected_direction",
                "selection_rank",
                "token",
                "total_count",
                "log_likelihood",
                "log_ratio_direct_vs_group",
            ]
        ].to_string(index=False)
    )

    direct = load_corpus(
        direct_path,
        "direct",
        args.text_column,
        args.timestamp_column,
        selected_markers,
        args.case_insensitive,
    )
    group = load_corpus(
        group_path,
        "group",
        args.text_column,
        args.timestamp_column,
        selected_markers,
        args.case_insensitive,
    )
    combined = pd.concat([direct, group], ignore_index=True)

    selection_table.to_csv(
        tables_dir / "top_keyness_token_selection.csv",
        index=False,
        encoding="utf-8",
    )

    # The main keyness figure must use the normalized frequencies from the
    # same analysis that produced the keyness ranking.
    keyness_figure_source = build_keyness_figure_source(
        selection_table
    )
    keyness_figure_source.to_csv(
        tables_dir / "top_keyness_direct_group_figure_source.csv",
        index=False,
        encoding="utf-8",
    )

    direct_classified, direct_cells, direct_meta = classify_volume_cells(
        direct, "direct", args.low_quantile, args.high_quantile
    )
    group_classified, group_cells, group_meta = classify_volume_cells(
        group, "group", args.low_quantile, args.high_quantile
    )
    pd.concat([direct_cells, group_cells], ignore_index=True).to_csv(
        tables_dir / "weekday_hour_volume_classification.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame([direct_meta, group_meta]).to_csv(
        tables_dir / "weekday_hour_volume_thresholds.csv",
        index=False,
        encoding="utf-8",
    )

    volume_table = build_volume_marker_table(
        [direct_classified, group_classified],
        selected_markers,
    )
    volume_table.to_csv(
        tables_dir / "selected_markers_by_volume_class.csv",
        index=False,
        encoding="utf-8",
    )

    plot_top_keyness_dumbbell(
        keyness_figure_source,
        figures_dir / "top_keyness_tokens_direct_vs_group",
        args.dpi,
        args.with_internal_titles,
    )

    marker_order = (
        selection_table.sort_values(
            ["selected_direction", "selection_rank"],
            ascending=[True, True],
        )["token"]
        .drop_duplicates()
        .tolist()
    )


    plot_volume_difference(
        volume_table,
        marker_order,
        figures_dir / "top_keyness_tokens_low_minus_high_volume",
        args.dpi,
        args.with_internal_titles,
    )

    

    print("\nCreated figures:")
    print(figures_dir / "top_keyness_tokens_direct_vs_group.[png|svg|pdf]")
    print(figures_dir / "top_keyness_tokens_low_minus_high_volume.[png|svg|pdf]")
    print("\nCreated aggregated source tables:")
    for name in [
        "top_keyness_token_selection.csv",
        "top_keyness_direct_group_figure_source.csv",
        "weekday_hour_volume_classification.csv",
        "weekday_hour_volume_thresholds.csv",
        "selected_markers_by_volume_class.csv",
    ]:
        print(tables_dir / name)
    print("\nVolume classification summary:")
    print(pd.DataFrame([direct_meta, group_meta]).to_string(index=False))


if __name__ == "__main__":
    main()
