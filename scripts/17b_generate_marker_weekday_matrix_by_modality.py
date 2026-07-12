#!/usr/bin/env python3
"""
17b_generate_marker_weekday_matrix_by_modality.py

Create a publication-ready two-panel weekday matrix for selected workflow
markers, separated into direct and group messages.

Visual encoding:
- rows = workflow markers
- columns = weekdays (Mon-Sun)
- panel A = direct messages
- panel B = group messages
- square area = percentage of messages within that modality and weekday
  containing the marker at least once

Only aggregated public outputs are written. Message text is never exported.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DIRECT_COLOR = "#D9D9D9"
GROUP_COLOR = "#595959"
EDGE_COLOR = "#303030"
GRID_COLOR = "#E0E0E0"
PANEL_BG = "#F7F7F7"
WHITE = "#FFFFFF"
WEEKEND_BG = "#F1F1F1"

WEEKDAY_ORDER = list(range(7))
WEEKDAY_LABELS = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

DIRECTION_LABELS = {
    "direct": "Direct messages",
    "group": "Group messages",
}

DIRECTION_COLORS = {
    "direct": DIRECT_COLOR,
    "group": GROUP_COLOR,
}

DEFAULT_MARKERS = [
    "Übergabe",
    "WE",
    "kein_Todo",
    "Rückmeldung",
    "anwesend",
]

DISPLAY_LABELS = {
    "Übergabe": "Weekend handover",
    "WE": "Weekend preparation",
    "kein_Todo": "No active to-do",
    "Rückmeldung": "Therapy group feedback",
    "anwesend": "Therapy group attendance",
}

TOKEN_PATTERN = re.compile(
    r"[\wÄÖÜäöüß]+(?:_[\wÄÖÜäöüß]+)*",
    flags=re.UNICODE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a two-panel Mon-Sun term-by-time matrix for direct and "
            "group messages."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--direct-path",
        type=Path,
        default=Path(
            "outputs/confidential/cleaned_corpus_tables/"
            "D_utterances_clean_lexical_v2.csv"
        ),
        help="Final cleaned direct-message table.",
    )
    parser.add_argument(
        "--group-path",
        type=Path,
        default=Path(
            "outputs/confidential/cleaned_corpus_tables/"
            "G_utterances_clean_lexical_v2.csv"
        ),
        help="Final cleaned group-message table.",
    )
    parser.add_argument(
        "--text-column",
        default="text_clean_lexical_v2",
        help="Cleaned text column used for marker detection.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Unix timestamp column. Default: timestamp.",
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        default=DEFAULT_MARKERS,
        help="Exact normalized marker tokens to include.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/public/figures/accessible_results"),
        help="Directory for publication figures.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/public/tables/figure_sources"),
        help="Directory for aggregated figure-source tables.",
    )
    parser.add_argument(
        "--output-stem",
        default="marker_weekday_matrix_direct_group",
        help="Filename stem for figure outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600.",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Detect marker tokens case-insensitively.",
    )
    parser.add_argument(
        "--with-title",
        action="store_true",
        help="Include a shared title inside the figure.",
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_text_column(
    df: pd.DataFrame,
    requested_column: str,
    path: Path,
) -> str:
    if requested_column in df.columns:
        return requested_column

    for candidate in [
        "text_clean_lexical_v2",
        "interaction_text_v2",
        "text_clean_lexical",
        "text",
    ]:
        if candidate in df.columns:
            print(
                f"Requested text column '{requested_column}' not found in "
                f"{path.name}; using '{candidate}' instead."
            )
            return candidate

    raise ValueError(
        f"No supported text column found in {path}. "
        f"Available columns: {df.columns.tolist()}"
    )


def tokenize(text: object, case_insensitive: bool) -> list[str]:
    tokens = TOKEN_PATTERN.findall("" if pd.isna(text) else str(text))
    return [token.casefold() for token in tokens] if case_insensitive else tokens


def load_corpus(
    path: Path,
    direction: str,
    text_column: str,
    timestamp_column: str,
    markers: list[str],
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

    actual_text_column = resolve_text_column(df, text_column, path)

    out = pd.DataFrame(index=df.index)
    out["direction"] = direction
    out["text"] = df[actual_text_column].fillna("").astype(str)
    out["timestamp"] = pd.to_numeric(
        df[timestamp_column],
        errors="coerce",
    )
    out["datetime"] = pd.to_datetime(
        out["timestamp"],
        unit="s",
        errors="coerce",
    )
    out = out.dropna(subset=["datetime"]).copy()
    out["weekday"] = out["datetime"].dt.dayofweek.astype(int)

    token_lists = out["text"].map(
        lambda value: tokenize(value, case_insensitive)
    )

    marker_keys = [
        marker.casefold() if case_insensitive else marker
        for marker in markers
    ]

    for marker, marker_key in zip(markers, marker_keys):
        out[f"contains__{marker}"] = token_lists.map(
            lambda tokens, key=marker_key: key in tokens
        )

    if out.empty:
        raise ValueError(
            f"No valid timestamped messages loaded from {path}."
        )

    print(
        f"Loaded {len(out):,} {direction} messages with valid timestamps."
    )
    return out.reset_index(drop=True)


def build_denominator_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for direction in ["direct", "group"]:
        for weekday in WEEKDAY_ORDER:
            count = int(
                (
                    (combined["direction"] == direction)
                    & (combined["weekday"] == weekday)
                ).sum()
            )

            rows.append(
                {
                    "direction": direction,
                    "message_type": DIRECTION_LABELS[direction],
                    "weekday": weekday,
                    "weekday_label": WEEKDAY_LABELS[weekday],
                    "message_count": count,
                }
            )

    return pd.DataFrame(rows)


def build_source_table(
    combined: pd.DataFrame,
    markers: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for direction in ["direct", "group"]:
        direction_df = combined[
            combined["direction"] == direction
        ]

        for marker in markers:
            marker_column = f"contains__{marker}"

            for weekday in WEEKDAY_ORDER:
                sub = direction_df[
                    direction_df["weekday"] == weekday
                ]
                denominator = int(len(sub))
                messages_with_marker = int(
                    sub[marker_column].sum()
                )

                share = (
                    messages_with_marker / denominator * 100
                    if denominator > 0
                    else 0.0
                )

                rows.append(
                    {
                        "direction": direction,
                        "message_type": DIRECTION_LABELS[direction],
                        "marker": marker,
                        "display_label": DISPLAY_LABELS.get(
                            marker,
                            marker,
                        ),
                        "weekday": weekday,
                        "weekday_label": WEEKDAY_LABELS[weekday],
                        "messages_with_marker": messages_with_marker,
                        "message_denominator": denominator,
                        "message_share_percent": share,
                    }
                )

    return pd.DataFrame(rows)


def source_to_matrix(
    source: pd.DataFrame,
    direction: str,
    markers: list[str],
) -> pd.DataFrame:
    sub = source[source["direction"] == direction]

    matrix = sub.pivot(
        index="marker",
        columns="weekday",
        values="message_share_percent",
    )

    return matrix.reindex(
        index=markers,
        columns=WEEKDAY_ORDER,
        fill_value=0.0,
    )


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": EDGE_COLOR,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": EDGE_COLOR,
            "xtick.color": EDGE_COLOR,
            "ytick.color": EDGE_COLOR,
            "figure.facecolor": WHITE,
            "axes.facecolor": PANEL_BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(
    fig: plt.Figure,
    out_base: Path,
    dpi: int,
) -> None:
    out_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kwargs = {
        "bbox_inches": "tight",
        "facecolor": WHITE,
    }

    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=dpi,
        **kwargs,
    )
    fig.savefig(
        out_base.with_suffix(".svg"),
        **kwargs,
    )
    fig.savefig(
        out_base.with_suffix(".pdf"),
        **kwargs,
    )
    plt.close(fig)


def square_side_from_value(
    value: float,
    reference_max: float,
    max_side: float,
) -> float:
    """
    Return square side length with square area proportional to value.
    """
    if value <= 0 or reference_max <= 0:
        return 0.0

    return max_side * np.sqrt(value / reference_max)


def plot_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    denominator: pd.DataFrame,
    markers: list[str],
    direction: str,
    panel_label: str,
    scale_max: float,
) -> None:
    values = matrix.to_numpy(dtype=float)
    n_rows = len(markers)
    n_cols = len(WEEKDAY_ORDER)

    ax.set_facecolor(PANEL_BG)

    for weekday in [5, 6]:
        ax.axvspan(
            weekday - 0.5,
            weekday + 0.5,
            color=WEEKEND_BG,
            zorder=-2,
        )

    for x in np.arange(-0.5, n_cols + 0.5, 1):
        ax.axvline(
            x,
            color=GRID_COLOR,
            linewidth=0.8,
            zorder=0,
        )

    for y in np.arange(-0.5, n_rows + 0.5, 1):
        ax.axhline(
            y,
            color=GRID_COLOR,
            linewidth=0.8,
            zorder=0,
        )

    max_side = 0.66

    for row in range(n_rows):
        for column in range(n_cols):
            value = values[row, column]
            side = square_side_from_value(
                value=value,
                reference_max=scale_max,
                max_side=max_side,
            )

            if side <= 0:
                continue

            rectangle = Rectangle(
                (
                    column - side / 2,
                    row - side / 2,
                ),
                side,
                side,
                facecolor=DIRECTION_COLORS[direction],
                edgecolor=EDGE_COLOR,
                linewidth=0.7,
                zorder=3,
            )
            ax.add_patch(rectangle)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    denominator_lookup = (
        denominator[
            denominator["direction"] == direction
        ]
        .set_index("weekday")["message_count"]
    )

    x_labels = [
        f"{WEEKDAY_LABELS[weekday]}\n"
        f"n={int(denominator_lookup.get(weekday, 0))}"
        for weekday in WEEKDAY_ORDER
    ]

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        x_labels,
        linespacing=1.35,
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [
            DISPLAY_LABELS.get(marker, marker)
            for marker in markers
        ]
    )

    ax.tick_params(
        axis="x",
        pad=8,
        length=0,
    )
    ax.tick_params(
        axis="y",
        pad=8,
        length=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{panel_label}. {DIRECTION_LABELS[direction]}",
        loc="center",
        fontweight="bold",
        pad=12,
    )


def plot_matrix(
    source: pd.DataFrame,
    denominator: pd.DataFrame,
    markers: list[str],
    out_base: Path,
    dpi: int,
    with_title: bool,
) -> None:
    direct_matrix = source_to_matrix(
        source,
        "direct",
        markers,
    )
    group_matrix = source_to_matrix(
        source,
        "group",
        markers,
    )

    all_values = np.concatenate(
        [
            direct_matrix.to_numpy(dtype=float).ravel(),
            group_matrix.to_numpy(dtype=float).ravel(),
        ]
    )
    observed_max = float(np.nanmax(all_values)) if all_values.size else 0.0
    scale_max = max(
        20.0,
        np.ceil(observed_max / 5.0) * 5.0,
    )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(13.2, 6.5),
        sharey=True,
        constrained_layout=False,
    )

    plot_panel(
        axes[0],
        direct_matrix,
        denominator,
        markers,
        "direct",
        "A",
        scale_max,
    )
    plot_panel(
        axes[1],
        group_matrix,
        denominator,
        markers,
        "group",
        "B",
        scale_max,
    )

    axes[1].tick_params(labelleft=False)

    if with_title:
        fig.suptitle(
            "Weekday distribution of selected workflow markers",
            x=0.5,
            y=0.965,
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    # Shared size legend.
    legend_values = [5.0, 10.0, 20.0]
    legend_y = 0.095
    legend_x_positions = [0.44, 0.52, 0.61]
    max_side_points = 18.0

    fig.text(
        0.23,
        legend_y,
        "Share of messages containing marker",
        ha="left",
        va="center",
        fontsize=9,
        color=EDGE_COLOR,
    )

    for x_position, value in zip(
        legend_x_positions,
        legend_values,
    ):
        marker_size = (
            max_side_points
            * np.sqrt(value / scale_max)
        ) ** 2

        fig.text(
            x_position,
            legend_y,
            "■",
            ha="center",
            va="center",
            fontsize=np.sqrt(marker_size),
            color=GROUP_COLOR,
        )
        fig.text(
            x_position,
            legend_y - 0.035,
            f"{value:.0f}%",
            ha="center",
            va="top",
            fontsize=8.5,
            color=EDGE_COLOR,
        )

    fig.text(
        0.5,
        0.025,
        "Weekend estimates are based on smaller modality-specific message "
        "denominators and should be interpreted cautiously.",
        ha="center",
        va="center",
        fontsize=8.7,
        color=EDGE_COLOR,
    )

    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.22,
        top=0.88,
        wspace=0.16,
    )

    save_figure(
        fig,
        out_base,
        dpi,
    )


def main() -> None:
    args = parse_args()
    set_publication_style()

    root = args.project_root.resolve()
    direct_path = resolve_path(
        root,
        args.direct_path,
    )
    group_path = resolve_path(
        root,
        args.group_path,
    )
    figures_dir = resolve_path(
        root,
        args.figures_dir,
    )
    tables_dir = resolve_path(
        root,
        args.tables_dir,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    direct = load_corpus(
        direct_path,
        "direct",
        args.text_column,
        args.timestamp_column,
        args.markers,
        args.case_insensitive,
    )
    group = load_corpus(
        group_path,
        "group",
        args.text_column,
        args.timestamp_column,
        args.markers,
        args.case_insensitive,
    )

    combined = pd.concat(
        [direct, group],
        ignore_index=True,
    )

    denominator = build_denominator_table(
        combined
    )
    source = build_source_table(
        combined,
        args.markers,
    )

    denominator_path = (
        tables_dir
        / "marker_weekday_denominators_by_modality.csv"
    )
    source_path = (
        tables_dir
        / "marker_weekday_matrix_by_modality_source.csv"
    )

    denominator.to_csv(
        denominator_path,
        index=False,
        encoding="utf-8",
    )
    source.to_csv(
        source_path,
        index=False,
        encoding="utf-8",
    )

    out_base = (
        figures_dir
        / args.output_stem
    )

    plot_matrix(
        source,
        denominator,
        args.markers,
        out_base,
        args.dpi,
        args.with_title,
    )

    print("\nCreated figure:")
    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".svg"))
    print(out_base.with_suffix(".pdf"))

    print("\nCreated source tables:")
    print(source_path)
    print(denominator_path)

    print("\nModality-specific weekday denominators:")
    print(denominator.to_string(index=False))


if __name__ == "__main__":
    main()
