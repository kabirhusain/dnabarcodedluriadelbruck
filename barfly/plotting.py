"""Shared plot style: the seaborn context and the colours used across all figures."""

import seaborn as sns
import palettable as pal

colorTable = {
    'k': [0, 0, 0],
    'g': [27 / 255, 158 / 255, 119 / 255],
    'o': [217 / 255, 95 / 255, 2 / 255],
    'b': [117 / 255, 112 / 255, 179 / 255],
}

# Expt 2: one sequential colour map per mismatch-repair strain, so that a strain keeps
# its hue across panels while bin count is encoded by shade.
STRAIN_COLORMAPS = {
    "yKH88": pal.colorbrewer.sequential.Blues_7.mpl_colormap,    # wildtype
    "yFR1": pal.colorbrewer.sequential.Greens_7.mpl_colormap,    # msh6D
    "yFR2": pal.colorbrewer.sequential.YlOrRd_7.mpl_colormap,    # msh2D
}

STRAIN_LABELS = {
    "yKH88": "BY4741",
    "yFR1": r"BY4741 $msh6\Delta$",
    "yFR2": r"BY4741 $msh2\Delta$",
}

STRAIN_NAMES = {
    "yKH88": "wildtype",
    "yFR1": r"msh6$\Delta$",
    "yFR2": r"msh2$\Delta$",
}


def use_paper_style():
    """Apply the seaborn context and grid style used for every panel."""
    sns.set_context("talk", rc={"font.size": 15,
                               "axes.titlesize": 15,
                               "axes.labelsize": 15})
    sns.set_style("whitegrid", {"grid.color": '.9',
                                "grid.linestyle": '--',
                                "axes.edgecolor": '.6',
                                "xtick.bottom": True,
                                "ytick.left": True})


def bin_count_colour(index, n_bin_counts, offset=4):
    """Shade for the i-th bin count on a shared sequential scale.

    The offset skips the palest end of the map, which is illegible on white.
    """
    return pal.colorbrewer.sequential.YlOrRd_7.mpl_colormap(
        (index + offset) / (n_bin_counts + offset))


def strain_bin_count_colour(strain, index, n_bin_counts, offset=4):
    """As `bin_count_colour`, within one strain's own colour map."""
    return STRAIN_COLORMAPS[strain]((index + offset) / (n_bin_counts + offset))
