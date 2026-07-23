"""Custom MatPlotlib formatting."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

if TYPE_CHECKING:
    from collections.abc import Sequence

PALETTE = {
    "hero": "#3D6FE5",  # your model
    "hero_light": "#A6C0F5",  # a sibling model
    "neutral_1": "#868C92",  # competitor
    "neutral_2": "#C4C9CE",  # competitor
    "accent_gold": "#E3A23C",  # competitor
    "accent_tan": "#ECD3A6",  # competitor
}

# default left-to-right color order for a series list
DEFAULT_CYCLE = [
    PALETTE["hero"],
    PALETTE["hero_light"],
    PALETTE["neutral_1"],
    PALETTE["neutral_2"],
    PALETTE["accent_gold"],
    PALETTE["accent_tan"],
]

_LABEL_DARK = "#202124"  # hero value labels
_LABEL_MUTE = "#5F6368"  # competitor value labels
_SUBLABEL = "#80868B"  # italic metric sub-labels


@dataclass
class Series:
    """One model / bar-series in the grouped chart."""

    name: str
    values: Sequence[float]
    color: str | None = None  # falls back to DEFAULT_CYCLE by position
    hero: bool = False  # highlight: hatch + bold larger value labels


def grouped_hero_bar(  # noqa: PLR0913, ANN201, D417
    ax,  # noqa: ANN001
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    sublabels: Sequence[str] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    value_fmt: str = "{:.1f}",
    hatch: str = "////",
    group_width: float = 0.82,
    ylim: tuple[float, float] | None = None,
    legend: bool = True,
):
    """Draw a Deep-style grouped bar chart onto `ax`.

    Args:
    ----
    categories : group labels along the x-axis (e.g. benchmark names)
    series     : list of Series; mark exactly one (or more) with hero=True
    sublabels  : optional italic text under each category (e.g. metric names)
    value_fmt  : format string for the on-bar value labels
    hatch      : hatch pattern for hero bars ("////" = candy stripe)
    group_width: fraction of the x-slot occupied by a full group of bars

    Returns:
    -------
        A Matplotlib axis

    """
    n_groups = len(categories)
    n_series = len(series)
    x = np.arange(n_groups)
    bar_w = group_width / n_series

    # value-label vertical padding = small fraction of the data range
    all_vals = np.concatenate([np.asarray(s.values, float) for s in series])
    vmax = float(np.nanmax(all_vals))
    pad = 0.012 * vmax

    for i, s in enumerate(series):
        color = s.color or DEFAULT_CYCLE[i % len(DEFAULT_CYCLE)]
        offset = (i - (n_series - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            s.values,
            bar_w,
            label=s.name,
            color=color,
            zorder=3,
        )

        if s.hero:
            # white candy-stripe hatch over the saturated fill.
            # a hairline white edge keeps the stripes crisp without a visible border.
            for b in bars:
                b.set_hatch(hatch)
                b.set_edgecolor("white")
                b.set_linewidth(0.4)

        # on-bar value labels
        for b, v in zip(bars, s.values, strict=False):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + pad,
                value_fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9 if s.hero else 7,
                fontweight="bold" if s.hero else "normal",
                color=_LABEL_DARK if s.hero else _LABEL_MUTE,
                zorder=4,
            )

    # x category labels (bold)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_xlim(-0.5, n_groups - 0.5)

    # italic metric sub-labels under each category
    if sublabels is not None:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for xi, sub in zip(x, sublabels, strict=False):
            ax.text(
                xi,
                -0.075,
                sub,
                transform=trans,
                ha="center",
                va="top",
                fontsize=7.5,
                style="italic",
                color=_SUBLABEL,
            )

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # headroom for the value labels
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, vmax * 1.12)

    if legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=n_series,
            handletextpad=0.5,
        )
    return ax


def _demo() -> None:
    plt.style.use("deep_hero.mplstyle")

    categories = ["MMLU-Pro", "GPQA-Diamond", "MATH-500", "AIME 2024", "Codeforces", "SWE-bench"]
    sublabels = ["EM", "Pass@1", "EM", "Pass@1", "Percentile", "Resolved"]

    series = [
        Series("Our-Model-V3", [75.9, 59.1, 90.2, 39.2, 51.6, 42.0], hero=True),
        Series("Our-Model-V2.5", [66.2, 41.3, 74.7, 16.7, 35.6, 22.6]),
        Series("Competitor-A", [71.6, 49.0, 80.0, 23.3, 24.8, 23.8]),
        Series("Competitor-B", [73.3, 51.1, 73.8, 23.3, 25.3, 24.5]),
        Series("Competitor-C", [72.6, 49.9, 74.6, 9.3, 23.6, 38.8]),
        Series("Competitor-D", [78.0, 65.0, 78.3, 16.0, 20.3, 50.8]),
    ]

    fig, ax = plt.subplots()
    grouped_hero_bar(
        ax,
        categories,
        series,
        sublabels=sublabels,
        ylabel="Accuracy / Percentile (%)",
        ylim=(0, 100),
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.07, right=0.99)
    fig.savefig("hero_bar_demo.png")
    print("wrote hero_bar_demo.png")  # noqa: T201


if __name__ == "__main__":
    _demo()
