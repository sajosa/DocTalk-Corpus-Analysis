#!/usr/bin/env python3
"""
13_time_analysis.py

Reproducible descriptive time analysis for the DocTalk corpus.

The script recreates the notebook-based time analyses using the cleaned
utterance CSV files produced by the pipeline. It does not use message text and
therefore writes only aggregated, non-confidential outputs to outputs/public/.

Default inputs:
    outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv
    outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv

Default outputs:
    outputs/public/tables/time/time_analysis_tables.xlsx
    outputs/public/tables/time/*.csv
    outputs/public/figures/time/*.png, *.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


DIRECT_COLOR = "#D9D9D9"
GROUP_COLOR = "#595959"
EDGE_COLOR = "#303030"
GRID_COLOR = "#E0E0E0"
PANEL_BG = "#F7F7F7"

DIRECTION_COLORS = {
    "direct": DIRECT_COLOR,
    "group": GROUP_COLOR,
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "doctalk_greys",
    [PANEL_BG, DIRECT_COLOR, GROUP_COLOR],
)


WEEKDAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

MESSAGE_LABELS = {
    "direct": "Direct messages",
    "group": "Group messages",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate public aggregated time-analysis tables and figures."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory. Default: current working directory.",
    )
    parser.add_argument(
        "--direct-path",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv"),
        help="Path to cleaned direct-message CSV, relative to project dir or absolute.",
    )
    parser.add_argument(
        "--group-path",
        type=Path,
        default=Path("outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv"),
        help="Path to cleaned group-message CSV, relative to project dir or absolute.",
    )
    parser.add_argument(
        "--out-tables-dir",
        type=Path,
        default=Path("outputs/public/tables/time"),
        help="Output directory for public time tables.",
    )
    parser.add_argument(
        "--out-figures-dir",
        type=Path,
        default=Path("outputs/public/figures/time"),
        help="Output directory for public time figures.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column in the input CSVs. Default: timestamp.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV copies of the main tables.",
    )
    return parser.parse_args()


def resolve_path(project_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_dir / path

def load_timestamp_table(
    path: Path,
    direction: str,
    timestamp_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = pd.DataFrame(index=df.index)
    out["direction"] = direction
    out["message_type"] = MESSAGE_LABELS[direction]
    out["timestamp"] = pd.to_numeric(
        df[timestamp_col],
        errors="coerce",
    )

    out = out.dropna(subset=["timestamp"]).copy()
    out["timestamp"] = out["timestamp"].astype("int64")

    out["datetime"] = pd.to_datetime(
        out["timestamp"],
        unit="s",
        errors="coerce",
    )

    out = out.dropna(subset=["datetime"]).copy()
    out["weekday"] = out["datetime"].dt.dayofweek.astype(int)
    out["weekday_name"] = out["weekday"].map(WEEKDAY_LABELS)
    out["hour"] = out["datetime"].dt.hour.astype(int)

    if out.empty:
        raise ValueError(
            f"No valid timestamps could be loaded from {path}."
        )

    print(
        f"Loaded {len(out):,} valid timestamps from {path.name} "
        f"for {direction} messages."
    )

    return out.reset_index(drop=True)


def complete_hourly_distribution(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_hours = pd.DataFrame({"hour": range(24)})

    for direction, group in ts.groupby("direction", sort=False):
        counts = group.groupby("hour").size().reset_index(name="count")
        full = all_hours.merge(counts, on="hour", how="left")
        full["count"] = full["count"].fillna(0).astype(int)
        total = int(full["count"].sum())
        full["relative_frequency"] = (full["count"] / total * 100) if total else 0
        full["direction"] = direction
        full["message_type"] = MESSAGE_LABELS.get(direction, direction)
        full["hour_interval"] = full["hour"].apply(lambda h: f"{h}:00–{h + 1}:00")
        rows.append(full[["direction", "message_type", "hour", "hour_interval", "count", "relative_frequency"]])

    if not rows:
        raise ValueError(
            "Hourly distribution could not be created because no valid "
            "timestamp records were available."
        )

    return pd.concat(rows, ignore_index=True)


def complete_weekday_distribution(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_weekdays = pd.DataFrame({"weekday": range(7)})

    for direction, group in ts.groupby("direction", sort=False):
        counts = group.groupby("weekday").size().reset_index(name="count")
        full = all_weekdays.merge(counts, on="weekday", how="left")
        full["count"] = full["count"].fillna(0).astype(int)
        total = int(full["count"].sum())
        full["relative_frequency"] = (full["count"] / total * 100) if total else 0
        full["direction"] = direction
        full["message_type"] = MESSAGE_LABELS.get(direction, direction)
        full["weekday_name"] = full["weekday"].map(WEEKDAY_LABELS)
        rows.append(full[["direction", "message_type", "weekday", "weekday_name", "count", "relative_frequency"]])

    if not rows:
        raise ValueError(
            "Weekday distribution could not be created because no valid "
            "timestamp records were available."
        )

    return pd.concat(rows, ignore_index=True)


def complete_weekday_hour_distribution(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_grid = pd.MultiIndex.from_product(
        [range(7), range(24)], names=["weekday", "hour"]
    ).to_frame(index=False)

    for direction, group in ts.groupby("direction", sort=False):
        counts = group.groupby(["weekday", "hour"]).size().reset_index(name="count")
        full = full_grid.merge(counts, on=["weekday", "hour"], how="left")
        full["count"] = full["count"].fillna(0).astype(int)
        total = int(full["count"].sum())
        full["relative_frequency"] = (full["count"] / total * 100) if total else 0
        full["direction"] = direction
        full["message_type"] = MESSAGE_LABELS.get(direction, direction)
        full["weekday_name"] = full["weekday"].map(WEEKDAY_LABELS)
        rows.append(full[["direction", "message_type", "weekday", "weekday_name", "hour", "count", "relative_frequency"]])

    if not rows:
        raise ValueError(
            "Weekday-hour distribution could not be created because no valid "
            "timestamp records were available."
        )

    return pd.concat(rows, ignore_index=True)


def create_summary(ts: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for direction, group in ts.groupby("direction", sort=False):
        summary.append({
            "direction": direction,
            "message_type": MESSAGE_LABELS.get(direction, direction),
            "messages_with_valid_timestamp": len(group),
            "min_datetime": group["datetime"].min(),
            "max_datetime": group["datetime"].max(),
            "busiest_hour": int(group["hour"].value_counts().idxmax()) if len(group) else pd.NA,
            "busiest_weekday": WEEKDAY_LABELS[int(group["weekday"].value_counts().idxmax())] if len(group) else pd.NA,
        })
    return pd.DataFrame(summary)


def pivot_matrix(weekday_hour_df: pd.DataFrame, direction: str, value_col: str = "relative_frequency") -> pd.DataFrame:
    sub = weekday_hour_df[weekday_hour_df["direction"] == direction].copy()
    matrix = (
        sub.pivot(index="weekday_name", columns="hour", values=value_col)
        .reindex(list(WEEKDAY_LABELS.values()))
    )
    return matrix


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save publication figures in raster and vector formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    fig.savefig(out_dir / f"{stem}.png", dpi=300, **save_kwargs)
    fig.savefig(out_dir / f"{stem}.svg", **save_kwargs)
    fig.savefig(out_dir / f"{stem}.pdf", **save_kwargs)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    """Apply the shared publication style to a Matplotlib axis."""
    ax.set_facecolor(PANEL_BG)
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle="-",
        linewidth=0.8,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.tick_params(colors=EDGE_COLOR)
    ax.xaxis.label.set_color(EDGE_COLOR)
    ax.yaxis.label.set_color(EDGE_COLOR)
    ax.title.set_color(EDGE_COLOR)


def plot_hourly(hourly: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    style_axis(ax)

    for direction in ["direct", "group"]:
        sub = hourly[hourly["direction"] == direction].sort_values("hour")
        if sub.empty:
            continue

        ax.plot(
            sub["hour"],
            sub["relative_frequency"],
            color=DIRECTION_COLORS[direction],
            marker="o",
            markerfacecolor=DIRECTION_COLORS[direction],
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.7,
            linewidth=2.0,
            markersize=4.5,
            label=MESSAGE_LABELS[direction],
            zorder=3,
        )

    ax.set_title("Message distribution by hour of day", loc="left", fontweight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Messages within corpus (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "messages_by_hour_of_day")


def plot_weekday(weekday: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    style_axis(ax)

    width = 0.38
    x = np.arange(7)
    directions = [
        direction
        for direction in ["direct", "group"]
        if direction in set(weekday["direction"])
    ]

    for i, direction in enumerate(directions):
        sub = weekday[weekday["direction"] == direction].sort_values("weekday")
        offset = (i - (len(directions) - 1) / 2) * width

        ax.bar(
            x + offset,
            sub["relative_frequency"],
            width=width,
            color=DIRECTION_COLORS[direction],
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            label=MESSAGE_LABELS[direction],
            zorder=3,
        )

    ax.set_title("Message distribution by weekday", loc="left", fontweight="bold")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Messages within corpus (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(WEEKDAY_LABELS.values()), rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "messages_by_weekday")


def plot_weekday_hour_heatmap(
    weekday_hour: pd.DataFrame,
    out_dir: Path,
) -> None:
    directions = [
        direction
        for direction in ["direct", "group"]
        if direction in set(weekday_hour["direction"])
    ]
    if not directions:
        raise ValueError(
            "Heatmap could not be created because no direction data were available."
        )

    matrices = {
        direction: pivot_matrix(weekday_hour, direction)
        for direction in directions
    }
    vmax = max(float(matrix.max().max()) for matrix in matrices.values())
    vmax = np.ceil(vmax * 2) / 2 if vmax > 0 else 1

    fig, axes = plt.subplots(
        nrows=len(directions),
        ncols=1,
        figsize=(8.5, 3.2 * len(directions)),
        sharex=True,
        constrained_layout=True,
    )
    if len(directions) == 1:
        axes = [axes]

    im = None
    for idx, direction in enumerate(directions):
        ax = axes[idx]
        matrix = matrices[direction]
        ax.set_facecolor(PANEL_BG)

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=HEATMAP_CMAP,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

        panel = chr(ord("A") + idx)
        ax.set_title(
            f"{panel}. {MESSAGE_LABELS[direction]}",
            loc="left",
            fontsize=11,
            fontweight="bold",
            color=EDGE_COLOR,
        )
        ax.set_ylabel("Weekday", color=EDGE_COLOR)
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(list(WEEKDAY_LABELS.values()), fontsize=9)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticklabels(
            [f"{hour:02d}:00" for hour in range(0, 24, 2)],
            fontsize=8,
        )
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(
            which="minor",
            color="white",
            linestyle="-",
            linewidth=0.45,
        )
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(colors=EDGE_COLOR)

        for spine in ax.spines.values():
            spine.set_color(EDGE_COLOR)
            spine.set_linewidth(0.7)

    axes[-1].set_xlabel("Hour of day", color=EDGE_COLOR)

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes,
            orientation="vertical",
            fraction=0.028,
            pad=0.025,
        )
        cbar.set_label("Messages within corpus (%)", fontsize=9, color=EDGE_COLOR)
        cbar.ax.tick_params(labelsize=8, colors=EDGE_COLOR)
        cbar.outline.set_edgecolor(EDGE_COLOR)

    save_figure(
        fig,
        out_dir,
        "combined_direct_group_weekday_hour_heatmap",
    )


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    direct_path = resolve_path(project_dir, args.direct_path)
    group_path = resolve_path(project_dir, args.group_path)
    out_tables_dir = resolve_path(project_dir, args.out_tables_dir)
    out_figures_dir = resolve_path(project_dir, args.out_figures_dir)

    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    direct_ts = load_timestamp_table(direct_path, "direct", args.timestamp_col)
    group_ts = load_timestamp_table(group_path, "group", args.timestamp_col)
    ts = pd.concat([direct_ts, group_ts], ignore_index=True)

    if ts.empty:
        raise ValueError(
            "No valid timestamp records were loaded from either corpus."
        )

    summary = create_summary(ts)
    hourly = complete_hourly_distribution(ts)
    weekday = complete_weekday_distribution(ts)
    weekday_hour = complete_weekday_hour_distribution(ts)

    xlsx_path = out_tables_dir / "time_analysis_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        hourly.to_excel(writer, sheet_name="hourly_distribution", index=False)
        weekday.to_excel(writer, sheet_name="weekday_distribution", index=False)
        weekday_hour.to_excel(writer, sheet_name="weekday_hour_distribution", index=False)

    if args.write_csv:
        summary.to_csv(out_tables_dir / "time_analysis_summary.csv", index=False, encoding="utf-8")
        hourly.to_csv(out_tables_dir / "combined_messages_hourly_distribution.csv", index=False, encoding="utf-8")
        weekday.to_csv(out_tables_dir / "combined_messages_weekday_distribution.csv", index=False, encoding="utf-8")
        weekday_hour.to_csv(out_tables_dir / "combined_messages_weekday_hour_distribution.csv", index=False, encoding="utf-8")

    plot_hourly(hourly, out_figures_dir)
    plot_weekday(weekday, out_figures_dir)
    plot_weekday_hour_heatmap(weekday_hour, out_figures_dir)

    print("Saved time analysis workbook:")
    print(xlsx_path)
    print("Saved time figures to:")
    print(out_figures_dir)
    print("Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
