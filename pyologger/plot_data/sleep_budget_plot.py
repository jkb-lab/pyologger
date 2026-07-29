"""
Hierarchical sleep-budget bar chart (matplotlib).

Layout (top → bottom):
  ┌─────────────────────────────────────────┐
  │  [Group A ████████]  [Group B ████████] │  ← thick-bordered, centered
  ├─────────────────────────────────────────┤
  │   [Subgroup A1 ██████]                  │  ← thin-bordered, one step left
  │     ind (no border, leftmost)  NES-1003 │
  │   [Subgroup A2 ████████████]            │
  │     ind …                               │
  ├─────────────────────────────────────────┤
  │   [Subgroup B1 ███████████]             │
  │     ind …                               │
  └─────────────────────────────────────────┘

Bar x-start is indented by level: group bars start furthest right,
subgroup one step left, individuals at x=0 (full-width reference).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# ── Visual hierarchy (index: 0=group, 1=subgroup, 2=individual) ───────────────
_BAR_HEIGHT   = [0.75, 0.50, 0.28]
_LABEL_FS     = [12,   9.5,  7.5]
_LABEL_FW     = ["bold", "semibold", "normal"]
_LABEL_COLOR  = ["#111", "#333", "#666"]
_VALUE_FS     = [10,   8.5,  6.5]
_VALUE_COLOR  = ["#111", "#333", "#555"]

# x-axis indent fraction per level: group bars start at this fraction of xmax
# (so they appear "centered" / inset), subgroup slightly less, individuals at 0
_X_INDENT_FRAC = [0.0, 0.0, 0.0]   # all start at 0; visual hierarchy via bar height + border

# border style per level: (linewidth, edgecolor, zorder_boost)
_BORDER = [
    (2.2, "#222", 1),   # group: thick dark border
    (1.0, "#555", 1),   # subgroup: thin medium border
    (0.0, "none",  0),  # individual: no border
]

# gap before a row of each level (row units)
_GAP_BEFORE             = {0: 0.0,  1: 0.70, 2: 0.30}
_GAP_BETWEEN_SUBGROUPS  = 0.60
_GROUP_BAR_SPACING      = 1.6    # vertical gap between adjacent group summary rows
_GAP_GROUP_TO_FIRST_SG  = 1.20  # gap from last group row to first subgroup

# left-label indent per level (pts rightward from left axes edge)
_LABEL_INDENT_PT = [0, 10, 22]


def _mean_row(rows: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    return {c: float(rows[c].mean()) for c in cols}


def daily_summary_comparison_plot(
    df: pd.DataFrame,
    groups: List[Dict[str, Any]],
    stack_cols: Sequence[str],
    stack_colors: Optional[Dict[str, str]] = None,
    stack_labels: Optional[Dict[str, str]] = None,
    xmax: Optional[float] = None,
    xlabel: str = "Sleep (h per 24 h)",
    id_col: str = "animal",
    left_margin: float = 0.20,
    right_margin: float = 0.10,
    row_height_in: float = 0.34,
    figsize_w: float = 12.0,
    show_values: bool = True,
    value_fmt: str = "{:.2f} h",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw a hierarchical horizontal stacked-bar figure.

    Group summary bars appear at the top (one per group, stacked vertically),
    bordered with a thick outline.  Subgroup bars follow below with a thin
    border.  Individual bars are borderless at the left edge.
    """
    colors = dict(stack_colors or {})
    leg_labels = {c: c.replace("_", " ").title() for c in stack_cols}
    leg_labels.update(stack_labels or {})

    if xmax is None:
        xmax = float(df[list(stack_cols)].sum(axis=1).max()) * 1.25

    # ── Build flat draw list with y positions ─────────────────────────────────
    # (level, label, data_df, ind_id_or_None, y_pos)
    draw: List[Tuple[int, str, pd.DataFrame, Optional[str], float]] = []

    # Group summaries: stacked at top, each separated by _GROUP_BAR_SPACING
    group_y: List[float] = []
    for gi, g in enumerate(groups):
        group_y.append(float(gi) * _GROUP_BAR_SPACING)
        draw.append((0, g["label"], g["rows"], None, group_y[-1]))

    # Children: all subgroups + individuals below the group band
    y = max(group_y) + _GAP_GROUP_TO_FIRST_SG

    for gi, g in enumerate(groups):
        for si, sg in enumerate(g.get("subgroups", [])):
            if si > 0:
                y += _GAP_BETWEEN_SUBGROUPS
            draw.append((1, sg["label"], sg["rows"], None, y))
            y += _GAP_BEFORE[2]
            for _, row in sg["rows"].sort_values(stack_cols[0]).iterrows():
                ind_id = str(row[id_col]) if id_col in row.index else str(row.name)
                draw.append((2, "", pd.DataFrame([row]), ind_id, y))
                y += _GAP_BEFORE[2]
        if gi < len(groups) - 1:
            y += _GAP_BETWEEN_SUBGROUPS * 1.5

    total_height = y + 1.0

    # ── Figure / axes ─────────────────────────────────────────────────────────
    if ax is None:
        fig_h = max(4.0, total_height * row_height_in + 1.5)
        fig, ax = plt.subplots(figsize=(figsize_w, fig_h))
    else:
        fig = ax.get_figure()

    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.8, total_height)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.grid(axis="x", linestyle="--", alpha=0.28, zorder=0)

    fig.subplots_adjust(left=left_margin, right=1.0 - right_margin)

    # Dashed separator between group band and children — sits halfway in the gap
    sep_y = max(group_y) + _GROUP_BAR_SPACING * 0.5 + 0.1
    ax.axhline(sep_y, color="#bbb", lw=0.8, linestyle="--", zorder=1)

    # ── Draw bars + annotations ───────────────────────────────────────────────
    ind_annotations: List[Tuple[float, str]] = []

    for level, label, rows, ind_id, yp in draw:
        bh    = _BAR_HEIGHT[level]
        means = _mean_row(rows, stack_cols)
        lw, ec, _ = _BORDER[level]

        x_left = 0.0
        for col in stack_cols:
            val = means.get(col, 0.0)
            if val > 0:
                ax.barh(yp, val, bh, left=x_left,
                        color=colors.get(col, "#aaa"), zorder=3,
                        edgecolor="none", linewidth=0)
                x_left += val

        total = sum(means.get(c, 0.0) for c in stack_cols)

        # Draw outline rectangle over the whole stacked bar.
        # barh centres bar on yp; Rectangle needs bottom-left corner.
        # With inverted y-axis, "bottom" in data coords = yp - bh/2.
        if lw > 0 and total > 0:
            rect = plt.Rectangle(
                (0, yp - bh / 2), total, bh,
                linewidth=lw, edgecolor=ec,
                facecolor="none", zorder=5,
                clip_on=False,
            )
            ax.add_patch(rect)

        if show_values and total > 0:
            ax.text(total + xmax * 0.012, yp,
                    value_fmt.format(total),
                    va="center", ha="left",
                    fontsize=_VALUE_FS[level],
                    color=_VALUE_COLOR[level])

        if level < 2:
            n   = len(rows)
            lbl = f"{label}  n={n}"
            indent_pt = _LABEL_INDENT_PT[level]
            # compute offset in points from left axes edge
            ax_w_pts = ax.get_position().width * fig.get_figwidth() * 72
            offset_pts = -(left_margin * ax_w_pts) + indent_pt
            ax.annotate(
                lbl,
                xy=(0, yp),
                xytext=(offset_pts, 0),
                xycoords=("axes fraction", "data"),
                textcoords="offset points",
                va="center", ha="left",
                fontsize=_LABEL_FS[level],
                fontweight=_LABEL_FW[level],
                color=_LABEL_COLOR[level],
                annotation_clip=False,
            )
        else:
            ind_annotations.append((yp, ind_id or ""))

    # Right-side individual IDs
    for yp, id_str in ind_annotations:
        ax.annotate(
            id_str,
            xy=(1, yp),
            xytext=(4, 0),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            va="center", ha="left",
            fontsize=_LABEL_FS[2],
            color=_LABEL_COLOR[2],
            annotation_clip=False,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=colors[c], label=leg_labels[c])
               for c in stack_cols if c in colors]
    ax.legend(handles=patches, loc="lower right", frameon=False,
              fontsize=8, bbox_to_anchor=(1.0, -0.06))

    return fig, ax
