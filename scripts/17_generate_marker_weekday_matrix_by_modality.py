#!/usr/bin/env python3
"""
17_generate_marker_weekday_matrix_by_modality.py

Create a publication-ready two-panel weekday matrix for selected workflow
markers, separated into direct and group messages.

Visual encoding
---------------
- rows: selected normalized workflow markers
- columns: weekdays from Monday to Sunday
- panel A: direct messages
- panel B: group messages
- square area: percentage of messages within the respective communication
  modality and weekday containing the marker at least once

Methodological properties
-------------------------
- The unit of analysis is the message/utterance.
- Repeated occurrences of the same marker within one message are counted once.
- Percentages use separate modality- and weekday-specific denominators.
- Both panels use the same square-area scale.
- The validated weekday column is preferred when available.
- Only aggregated public outputs are written.
- Original message text is never exported.

Example
-------
python scripts/17_generate_marker_weekday_matrix_by_modality.py \
    --project-root . \
    --validate-weekday-from-timestamp
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

MARKER_COLOR = "#666666"
EDGE_COLOR = "#303030"
TEXT_COLOR = "#303030"
GRID_COLOR = "#DEDEDE"
PANEL_BG = "#F7F7F7"
WEEKEND_BG = "#EEEEEE"
WHITE = "#FFFFFF"

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

DEFAULT_MARKERS = [
    "Übergabe",
    "WE",
    "kein_Todo",
    "Rückmeldung",
    "anwesend",
]

DISPLAY_LABELS = {
    "Übergabe": "Handover",
    "WE": "Weekend (WE)",
    "kein_Todo": "No active to-do",
    "Rückmeldung": "Therapy group feedback",
    "anwesend": "Therapy group attendance",
}

TOKEN_PATTERN = re.compile(
    r"[\wÄÖÜäöüß]+(?:_[\wÄÖÜäöüß]+)*",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a two-panel weekday matrix for selected workflow "
            "markers in direct and group messages."
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
        "--weekday-column",
        default="weekday",
        help=(
            "Validated weekday column encoded as Monday=0 through Sunday=6. "
            "If unavailable, weekday is derived from the timestamp."
        ),
    )

    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Unix timestamp column used as fallback. Default: timestamp.",
    )

    parser.add_argument(
        "--timestamp-is-utc",
        action="store_true",
        help=(
            "Interpret Unix timestamps as UTC and convert them to the timezone "
            "specified by --timezone before deriving weekdays."
        ),
    )

    parser.add_argument(
        "--timezone",
        default="Europe/Berlin",
        help=(
            "Timezone used when --timestamp-is-utc is enabled. "
            "Default: Europe/Berlin."
        ),
    )

    parser.add_argument(
        "--validate-weekday-from-timestamp",
        action="store_true",
        help=(
            "When both weekday and timestamp columns are present, compare the "
            "validated weekday column with weekdays derived from timestamps."
        ),
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
        default=Path(
            "outputs/public/figures/accessible_results"
        ),
        help="Directory for publication figures.",
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(
            "outputs/public/tables/figure_sources"
        ),
        help="Directory for aggregated figure-source tables.",
    )

    parser.add_argument(
        "--output-stem",
        default="marker_weekday_matrix_direct_group",
        help="Filename stem for PNG, SVG, and PDF outputs.",
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
        help=(
            "Include a shared title inside the figure. "
            "For manuscript submission, omitting the internal title is usually "
            "preferable because the journal caption supplies the title."
        ),
    )

    parser.add_argument(
        "--scale-maximum",
        type=float,
        default=None,
        help=(
            "Optional fixed maximum percentage for square scaling. "
            "If omitted, the scale is determined from the observed maximum "
            "and rounded up to the next 5 percentage points, with a minimum "
            "of 20%%."
        ),
    )

    parser.add_argument(
        "--maximum-square-area",
        type=float,
        default=900.0,
        help=(
            "Maximum square area in points squared. "
            "Default: 900."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def resolve_path(root: Path, path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else root / path


def resolve_text_column(
    df: pd.DataFrame,
    requested_column: str,
    path: Path,
) -> str:
    """
    Return the requested text column or a documented fallback.
    """
    if requested_column in df.columns:
        return requested_column

    fallback_columns = [
        "text_clean_lexical_v2",
        "interaction_text_v2",
        "text_clean_lexical",
        "text",
    ]

    for candidate in fallback_columns:
        if candidate in df.columns:
            warnings.warn(
                f"Requested text column '{requested_column}' was not found "
                f"in {path.name}. Using fallback column '{candidate}'.",
                stacklevel=2,
            )
            return candidate

    raise ValueError(
        f"No supported text column found in {path}. "
        f"Available columns: {df.columns.tolist()}"
    )


def tokenize(
    text: object,
    case_insensitive: bool,
) -> list[str]:
    """
    Tokenize normalized lexical text while preserving underscore compounds.
    """
    value = "" if pd.isna(text) else str(text)
    tokens = TOKEN_PATTERN.findall(value)

    if case_insensitive:
        return [token.casefold() for token in tokens]

    return tokens


# ---------------------------------------------------------------------------
# Weekday handling
# ---------------------------------------------------------------------------

def derive_datetime_from_timestamp(
    timestamp: pd.Series,
    timestamp_is_utc: bool,
    timezone: str,
) -> pd.Series:
    """
    Convert Unix timestamps to datetimes.

    When timestamp_is_utc is False, timestamps are converted without timezone
    conversion. This preserves the convention used in pipelines where Unix
    seconds encode a fixed synthetic reference week as local wall-clock time.

    When timestamp_is_utc is True, timestamps are interpreted as UTC and then
    converted to the specified local timezone.
    """
    numeric_timestamp = pd.to_numeric(
        timestamp,
        errors="coerce",
    )

    if timestamp_is_utc:
        return (
            pd.to_datetime(
                numeric_timestamp,
                unit="s",
                errors="coerce",
                utc=True,
            )
            .dt.tz_convert(timezone)
        )

    return pd.to_datetime(
        numeric_timestamp,
        unit="s",
        errors="coerce",
    )


def validate_weekday_values(
    weekday: pd.Series,
    path: Path,
) -> pd.Series:
    """
    Validate and return weekday values encoded from 0 through 6.
    """
    numeric_weekday = pd.to_numeric(
        weekday,
        errors="coerce",
    )

    invalid_nonmissing = (
        numeric_weekday.notna()
        & ~numeric_weekday.isin(WEEKDAY_ORDER)
    )

    if invalid_nonmissing.any():
        invalid_values = sorted(
            numeric_weekday.loc[invalid_nonmissing]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Invalid weekday values in {path}: {invalid_values}. "
            "Expected Monday=0 through Sunday=6."
        )

    return numeric_weekday.astype("Int64")


def assign_weekday(
    df: pd.DataFrame,
    path: Path,
    weekday_column: str,
    timestamp_column: str,
    timestamp_is_utc: bool,
    timezone: str,
    validate_from_timestamp: bool,
) -> pd.DataFrame:
    """
    Assign weekday values.

    Priority:
    1. Use the validated weekday column when available.
    2. Otherwise derive weekday from the timestamp column.

    Optional validation compares independently populated weekday values with
    weekdays derived from timestamps. Values filled from timestamps are not
    treated as an independent validation source.
    """
    out = df.copy()

    has_weekday = weekday_column in out.columns
    has_timestamp = timestamp_column in out.columns

    if not has_weekday and not has_timestamp:
        raise ValueError(
            f"Neither weekday column '{weekday_column}' nor timestamp column "
            f"'{timestamp_column}' was found in {path}. "
            f"Available columns: {out.columns.tolist()}"
        )

    if has_weekday:
        original_weekday = validate_weekday_values(
            out[weekday_column],
            path,
        )

        # Preserve which rows contained an independently available weekday
        # before any timestamp-based filling.
        originally_available_mask = original_weekday.notna()
        out["weekday"] = original_weekday.copy()

        if out["weekday"].isna().any():
            missing_count = int(out["weekday"].isna().sum())

            if not has_timestamp:
                raise ValueError(
                    f"{missing_count:,} rows in {path.name} have missing "
                    "weekday values, and no timestamp fallback exists."
                )

            derived_datetime = derive_datetime_from_timestamp(
                out[timestamp_column],
                timestamp_is_utc,
                timezone,
            )
            derived_weekday = derived_datetime.dt.dayofweek.astype("Int64")

            fill_mask = out["weekday"].isna() & derived_weekday.notna()

            out.loc[
                fill_mask,
                "weekday",
            ] = derived_weekday.loc[fill_mask]

            warnings.warn(
                f"Filled {int(fill_mask.sum()):,} missing weekday values in "
                f"{path.name} from timestamps.",
                stacklevel=2,
            )

        if validate_from_timestamp:
            if not has_timestamp:
                raise ValueError(
                    "--validate-weekday-from-timestamp was requested, but "
                    f"timestamp column '{timestamp_column}' is missing in "
                    f"{path}."
                )

            derived_datetime = derive_datetime_from_timestamp(
                out[timestamp_column],
                timestamp_is_utc,
                timezone,
            )
            derived_weekday = derived_datetime.dt.dayofweek.astype("Int64")

            # Compare only values that existed independently before filling.
            comparison_mask = (
                originally_available_mask
                & derived_weekday.notna()
            )

            if not comparison_mask.any():
                warnings.warn(
                    "No independently populated weekday values were "
                    f"available in {path.name}. Weekdays were derived from "
                    "timestamps, but an independent timestamp-versus-weekday "
                    "validation could not be performed.",
                    stacklevel=2,
                )
            else:
                mismatch_mask = (
                    comparison_mask
                    & (original_weekday != derived_weekday)
                )

                mismatch_count = int(mismatch_mask.sum())

                if mismatch_count > 0:
                    mismatch_preview = pd.DataFrame(
                        {
                            "validated_weekday": original_weekday.loc[
                                mismatch_mask
                            ],
                            "derived_weekday": derived_weekday.loc[
                                mismatch_mask
                            ],
                            "timestamp": out.loc[
                                mismatch_mask,
                                timestamp_column,
                            ],
                        }
                    ).head(10)

                    raise ValueError(
                        f"Weekday validation failed for {mismatch_count:,} "
                        f"rows in {path.name}.\n"
                        f"First mismatches:\n"
                        f"{mismatch_preview.to_string(index=False)}"
                    )

                print(
                    f"Independent weekday validation passed for "
                    f"{path.name}: {int(comparison_mask.sum()):,} rows."
                )

    else:
        derived_datetime = derive_datetime_from_timestamp(
            out[timestamp_column],
            timestamp_is_utc,
            timezone,
        )

        out["weekday"] = (
            derived_datetime
            .dt.dayofweek
            .astype("Int64")
        )

        warnings.warn(
            f"Validated weekday column '{weekday_column}' was not found in "
            f"{path.name}. Weekdays were derived from timestamps.",
            stacklevel=2,
        )

    out = out.dropna(subset=["weekday"]).copy()
    out["weekday"] = out["weekday"].astype(int)

    return out


# ---------------------------------------------------------------------------
# Corpus loading and marker detection
# ---------------------------------------------------------------------------

def load_corpus(
    path: Path,
    direction: str,
    text_column: str,
    weekday_column: str,
    timestamp_column: str,
    markers: list[str],
    case_insensitive: bool,
    timestamp_is_utc: bool,
    timezone: str,
    validate_weekday_from_timestamp: bool,
) -> pd.DataFrame:
    """
    Load one cleaned corpus and create binary message-level marker indicators.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Input file contains no rows: {path}"
        )

    actual_text_column = resolve_text_column(
        df,
        text_column,
        path,
    )

    df = assign_weekday(
        df=df,
        path=path,
        weekday_column=weekday_column,
        timestamp_column=timestamp_column,
        timestamp_is_utc=timestamp_is_utc,
        timezone=timezone,
        validate_from_timestamp=validate_weekday_from_timestamp,
    )

    out = pd.DataFrame(index=df.index)

    out["direction"] = direction
    out["text"] = (
        df[actual_text_column]
        .fillna("")
        .astype(str)
    )
    out["weekday"] = df["weekday"].astype(int)

    token_lists = out["text"].map(
        lambda value: tokenize(
            value,
            case_insensitive,
        )
    )

    marker_keys = [
        marker.casefold()
        if case_insensitive
        else marker
        for marker in markers
    ]

    for marker, marker_key in zip(
        markers,
        marker_keys,
    ):
        marker_column = f"contains__{marker}"

        out[marker_column] = token_lists.map(
            lambda tokens, key=marker_key: key in tokens
        )

    if out.empty:
        raise ValueError(
            f"No valid messages loaded from {path}."
        )

    print(
        f"Loaded {len(out):,} {direction} messages "
        f"from {path.name}."
    )

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregated public source tables
# ---------------------------------------------------------------------------

def build_denominator_table(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create modality- and weekday-specific message denominators.
    """
    rows: list[dict[str, object]] = []

    for direction in ["direct", "group"]:
        direction_df = combined[
            combined["direction"] == direction
        ]

        for weekday in WEEKDAY_ORDER:
            message_count = int(
                (direction_df["weekday"] == weekday).sum()
            )

            rows.append(
                {
                    "direction": direction,
                    "message_type": DIRECTION_LABELS[direction],
                    "weekday": weekday,
                    "weekday_label": WEEKDAY_LABELS[weekday],
                    "message_count": message_count,
                }
            )

    return pd.DataFrame(rows)


def build_source_table(
    combined: pd.DataFrame,
    markers: list[str],
) -> pd.DataFrame:
    """
    Create the aggregated marker-by-modality-by-weekday figure source table.
    """
    rows: list[dict[str, object]] = []

    for direction in ["direct", "group"]:
        direction_df = combined[
            combined["direction"] == direction
        ]

        for marker in markers:
            marker_column = f"contains__{marker}"

            if marker_column not in direction_df.columns:
                raise ValueError(
                    f"Required marker column missing: {marker_column}"
                )

            for weekday in WEEKDAY_ORDER:
                sub = direction_df[
                    direction_df["weekday"] == weekday
                ]

                denominator = int(len(sub))
                messages_with_marker = int(
                    sub[marker_column].sum()
                )

                share_percent = (
                    messages_with_marker
                    / denominator
                    * 100
                    if denominator > 0
                    else np.nan
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
                        "message_share_percent": share_percent,
                    }
                )

    return pd.DataFrame(rows)


def source_to_matrix(
    source: pd.DataFrame,
    direction: str,
    markers: list[str],
) -> pd.DataFrame:
    """
    Pivot source data into a marker-by-weekday percentage matrix.
    """
    sub = source[
        source["direction"] == direction
    ]

    matrix = sub.pivot(
        index="marker",
        columns="weekday",
        values="message_share_percent",
    )

    return matrix.reindex(
        index=markers,
        columns=WEEKDAY_ORDER,
    )


# ---------------------------------------------------------------------------
# Plot styling and scaling
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """
    Apply publication-oriented Matplotlib defaults.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": TEXT_COLOR,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": EDGE_COLOR,
            "xtick.color": EDGE_COLOR,
            "ytick.color": EDGE_COLOR,
            "figure.facecolor": WHITE,
            "axes.facecolor": PANEL_BG,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def determine_scale_maximum(
    direct_matrix: pd.DataFrame,
    group_matrix: pd.DataFrame,
    requested_maximum: float | None,
) -> float:
    """
    Determine a common maximum percentage for both panels.
    """
    if requested_maximum is not None:
        if requested_maximum <= 0:
            raise ValueError(
                "--scale-maximum must be greater than zero."
            )
        return float(requested_maximum)

    all_values = np.concatenate(
        [
            direct_matrix.to_numpy(dtype=float).ravel(),
            group_matrix.to_numpy(dtype=float).ravel(),
        ]
    )

    finite_values = all_values[np.isfinite(all_values)]

    observed_maximum = (
        float(np.max(finite_values))
        if finite_values.size
        else 0.0
    )

    rounded_maximum = (
        np.ceil(observed_maximum / 5.0) * 5.0
        if observed_maximum > 0
        else 20.0
    )

    return float(max(20.0, rounded_maximum))


def square_area_from_value(
    value: float,
    scale_maximum: float,
    maximum_square_area: float,
) -> float:
    """
    Return square marker area in points squared.

    Matplotlib scatter's 's' argument is an area measure in points squared.
    Therefore, setting s proportional to the percentage ensures that visible
    square area is proportional to the represented percentage.
    """
    if (
        not np.isfinite(value)
        or value <= 0
        or scale_maximum <= 0
    ):
        return 0.0

    clipped_value = min(
        float(value),
        float(scale_maximum),
    )

    return (
        maximum_square_area
        * clipped_value
        / scale_maximum
    )


# ---------------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------------

def plot_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    denominator: pd.DataFrame,
    markers: list[str],
    direction: str,
    panel_label: str,
    scale_maximum: float,
    maximum_square_area: float,
) -> None:
    """
    Draw one modality panel.
    """
    n_rows = len(markers)
    n_columns = len(WEEKDAY_ORDER)

    ax.set_facecolor(PANEL_BG)

    # Subtle weekend background.
    for weekday in [5, 6]:
        ax.axvspan(
            weekday - 0.5,
            weekday + 0.5,
            color=WEEKEND_BG,
            zorder=-3,
        )

    # Cell grid.
    for x_position in np.arange(
        -0.5,
        n_columns + 0.5,
        1,
    ):
        ax.axvline(
            x_position,
            color=GRID_COLOR,
            linewidth=0.8,
            zorder=-1,
        )

    for y_position in np.arange(
        -0.5,
        n_rows + 0.5,
        1,
    ):
        ax.axhline(
            y_position,
            color=GRID_COLOR,
            linewidth=0.8,
            zorder=-1,
        )

    # Draw squares using scatter so that legend and panel use exactly the same
    # area transformation.
    for row_index, marker in enumerate(markers):
        for weekday in WEEKDAY_ORDER:
            value = matrix.loc[marker, weekday]

            marker_area = square_area_from_value(
                value=value,
                scale_maximum=scale_maximum,
                maximum_square_area=maximum_square_area,
            )

            if marker_area <= 0:
                continue

            ax.scatter(
                weekday,
                row_index,
                s=marker_area,
                marker="s",
                facecolor=MARKER_COLOR,
                edgecolor=EDGE_COLOR,
                linewidth=0.75,
                zorder=3,
            )

    ax.set_xlim(
        -0.5,
        n_columns - 0.5,
    )
    ax.set_ylim(
        n_rows - 0.5,
        -0.5,
    )

    denominator_lookup = (
        denominator[
            denominator["direction"] == direction
        ]
        .set_index("weekday")["message_count"]
    )

    x_labels = [
        (
            f"{WEEKDAY_LABELS[weekday]}\n"
            f"n={int(denominator_lookup.get(weekday, 0))}"
        )
        for weekday in WEEKDAY_ORDER
    ]

    ax.set_xticks(
        np.arange(n_columns)
    )
    ax.set_xticklabels(
        x_labels,
        linespacing=1.3,
    )

    ax.set_yticks(
        np.arange(n_rows)
    )
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
        pad=9,
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


def add_size_legend(
    fig: plt.Figure,
    scale_maximum: float,
    maximum_square_area: float,
) -> None:
    """
    Add a compact common square-area legend.

    Legend markers use the same square_area_from_value() function as the data
    panels, ensuring consistent area scaling. Explicit x positions keep the
    reference squares visually grouped without crowding their labels.
    """
    legend_values = [
        value
        for value in [5.0, 10.0, 20.0]
        if value <= scale_maximum
    ]

    if not legend_values:
        legend_values = [scale_maximum]

    # Compact spacing between the three reference squares.
    x_positions = np.arange(len(legend_values), dtype=float) * 0.72

    legend_axis = fig.add_axes(
        [0.31, 0.025, 0.38, 0.10]
    )

    legend_axis.set_xlim(
        -0.95,
        float(x_positions[-1]) + 0.38,
    )
    legend_axis.set_ylim(
        -0.55,
        0.65,
    )
    legend_axis.axis("off")

    legend_axis.text(
        -0.30,
        0.15,
        "Share of messages\ncontaining marker",
        ha="right",
        va="center",
        fontsize=9,
        color=TEXT_COLOR,
        linespacing=1.25,
    )

    for x_position, value in zip(
        x_positions,
        legend_values,
    ):
        marker_area = square_area_from_value(
            value=value,
            scale_maximum=scale_maximum,
            maximum_square_area=maximum_square_area,
        )

        legend_axis.scatter(
            x_position,
            0.22,
            s=marker_area,
            marker="s",
            facecolor=MARKER_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.75,
        )

        legend_axis.text(
            x_position,
            -0.27,
            f"{value:.0f}%",
            ha="center",
            va="top",
            fontsize=8.5,
            color=TEXT_COLOR,
        )


def save_figure(
    fig: plt.Figure,
    out_base: Path,
    dpi: int,
) -> None:
    """
    Save PNG, SVG, and PDF versions.
    """
    out_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_kwargs = {
        "bbox_inches": "tight",
        "facecolor": WHITE,
    }

    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=dpi,
        **save_kwargs,
    )

    fig.savefig(
        out_base.with_suffix(".svg"),
        **save_kwargs,
    )

    fig.savefig(
        out_base.with_suffix(".pdf"),
        **save_kwargs,
    )

    plt.close(fig)


def plot_matrix(
    source: pd.DataFrame,
    denominator: pd.DataFrame,
    markers: list[str],
    out_base: Path,
    dpi: int,
    with_title: bool,
    requested_scale_maximum: float | None,
    maximum_square_area: float,
) -> None:
    """
    Create and save the final two-panel figure.
    """
    direct_matrix = source_to_matrix(
        source=source,
        direction="direct",
        markers=markers,
    )

    group_matrix = source_to_matrix(
        source=source,
        direction="group",
        markers=markers,
    )

    scale_maximum = determine_scale_maximum(
        direct_matrix=direct_matrix,
        group_matrix=group_matrix,
        requested_maximum=requested_scale_maximum,
    )

    observed_maximum = float(
        np.nanmax(
            np.concatenate(
                [
                    direct_matrix.to_numpy(
                        dtype=float
                    ).ravel(),
                    group_matrix.to_numpy(
                        dtype=float
                    ).ravel(),
                ]
            )
        )
    )

    if observed_maximum > scale_maximum:
        warnings.warn(
            f"Observed maximum ({observed_maximum:.2f}%) exceeds the selected "
            f"scale maximum ({scale_maximum:.2f}%). Values above the maximum "
            "will be visually clipped.",
            stacklevel=2,
        )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(13.2, 6.35),
        sharey=True,
        constrained_layout=False,
    )

    plot_panel(
        ax=axes[0],
        matrix=direct_matrix,
        denominator=denominator,
        markers=markers,
        direction="direct",
        panel_label="A",
        scale_maximum=scale_maximum,
        maximum_square_area=maximum_square_area,
    )

    plot_panel(
        ax=axes[1],
        matrix=group_matrix,
        denominator=denominator,
        markers=markers,
        direction="group",
        panel_label="B",
        scale_maximum=scale_maximum,
        maximum_square_area=maximum_square_area,
    )

    axes[1].tick_params(
        labelleft=False
    )

    if with_title:
        fig.suptitle(
            "Weekday distribution of selected workflow markers",
            x=0.5,
            y=0.975,
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    add_size_legend(
        fig=fig,
        scale_maximum=scale_maximum,
        maximum_square_area=maximum_square_area,
    )

    fig.subplots_adjust(
        left=0.20,
        right=0.985,
        bottom=0.30,
        top=0.88 if with_title else 0.92,
        wspace=0.16,
    )

    save_figure(
        fig=fig,
        out_base=out_base,
        dpi=dpi,
    )

    print(
        f"Common square-area scale maximum: "
        f"{scale_maximum:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

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
        path=direct_path,
        direction="direct",
        text_column=args.text_column,
        weekday_column=args.weekday_column,
        timestamp_column=args.timestamp_column,
        markers=args.markers,
        case_insensitive=args.case_insensitive,
        timestamp_is_utc=args.timestamp_is_utc,
        timezone=args.timezone,
        validate_weekday_from_timestamp=(
            args.validate_weekday_from_timestamp
        ),
    )

    group = load_corpus(
        path=group_path,
        direction="group",
        text_column=args.text_column,
        weekday_column=args.weekday_column,
        timestamp_column=args.timestamp_column,
        markers=args.markers,
        case_insensitive=args.case_insensitive,
        timestamp_is_utc=args.timestamp_is_utc,
        timezone=args.timezone,
        validate_weekday_from_timestamp=(
            args.validate_weekday_from_timestamp
        ),
    )

    combined = pd.concat(
        [direct, group],
        ignore_index=True,
    )

    expected_total = len(direct) + len(group)

    if len(combined) != expected_total:
        raise RuntimeError(
            "Combined corpus row count does not equal the sum of direct "
            "and group rows."
        )

    denominator = build_denominator_table(
        combined
    )

    source = build_source_table(
        combined=combined,
        markers=args.markers,
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
        source=source,
        denominator=denominator,
        markers=args.markers,
        out_base=out_base,
        dpi=args.dpi,
        with_title=args.with_title,
        requested_scale_maximum=args.scale_maximum,
        maximum_square_area=args.maximum_square_area,
    )

    print("\nCorpus totals:")
    print(f"Direct messages: {len(direct):,}")
    print(f"Group messages:  {len(group):,}")
    print(f"Combined:        {len(combined):,}")

    print("\nCreated figures:")
    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".svg"))
    print(out_base.with_suffix(".pdf"))

    print("\nCreated aggregated source tables:")
    print(source_path)
    print(denominator_path)

    print("\nModality-specific weekday denominators:")
    print(
        denominator.to_string(
            index=False
        )
    )

    print("\nMaximum observed percentages by marker and modality:")
    summary = (
        source.groupby(
            [
                "message_type",
                "display_label",
            ],
            as_index=False,
        )["message_share_percent"]
        .max()
        .rename(
            columns={
                "message_share_percent":
                    "maximum_weekday_share_percent"
            }
        )
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


if __name__ == "__main__":
    main()