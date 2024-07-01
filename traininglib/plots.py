""" Training and testing specific plots. """
from typing import List, Dict, Iterable
import math
from pathlib import Path
from itertools import zip_longest

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ebop_maven.plotting import format_axes
from ebop_maven.libs.mistisochrones import MistIsochrones
from ebop_maven import deb_example

from .datasets import read_param_sets_from_csvs

all_pub_labels = {
    "rA_plus_rB":   r"$r_{A}+r_{B}$",
    "k":            r"$k$",
    "inc":          r"$i$",
    "J":            r"$J$",
    "ecosw":        r"$e\,\cos{\omega}$",
    "esinw":        r"$e\,\sin{\omega}$",
    "L3":           r"$L_{3}$",
    "bP":           r"$b_{P}$",
}

# The full set of parameters available for histograms, their #bins and plot labels
histogram_params = {
    "rA_plus_rB":   (100, r"$r_{A}+r_{B}$"),
    "k":            (100, r"$k$"),
    "inc":          (100, r"$i~(^{\circ})$"),
    "sini":         (100, r"$\sin{i}$"),
    "cosi":         (100, r"$\cos{i}$"),
    "qphot":        (100, r"$q_{phot}$"),
    #"L3":           (100, r"$L_3$"), # currently always zero
    "ecc":          (100, r"$e$"),
    "omega":        (100, r"$\omega~(^{\circ})$"),
    "J":            (100, r"$J$"),
    "ecosw":        (100, r"$e\,\cos{\omega}$"),
    "esinw":        (100, r"$e\,\sin{\omega}$"),
    "rA":           (100, r"$r_A$"),
    "rB":           (100, r"$r_B$"),
    "bP":           (100, r"$b_{prim}$")
}


def plot_trainset_histograms(trainset_dir: Path,
                             plot_file: Path=None,
                             params: List[str]=None,
                             cols: int=3,
                             yscale: str="log",
                             verbose: bool=True):
    """
    Saves histogram plots to a single figure on a grid of axes. The params will
    be plotted in the order they are listed, scanning from left to right and down.

    :trainset_dir: the directory containing the trainset csv files
    :plot_file: the directory to save the plots. If none, they're saved with the trainset
    :parameters: the list of parameters to plot, or the full list if None.
    See the histogram_parameters attribute for the full list
    :cols: the width of the axes grid (the rows automatically adjust)
    :yscale: set to "linear" or "log" to control the y-axis scale
    :verbose: whether to print verbose progress/diagnostic messages
    """
    # pylint: disable=too-many-arguments
    if not params:
        param_specs = histogram_params
    else:
        param_specs = { p: histogram_params[p] for p in params if p in histogram_params }
    csvs = sorted(trainset_dir.glob("trainset*.csv"))

    if param_specs and csvs:
        rows = math.ceil(len(param_specs) / cols)
        _, axes = plt.subplots(rows, cols, sharey="all",
                               figsize=(cols*3, rows*2.5), tight_layout=True)
        if verbose:
            print(f"Plotting histograms in a {cols}x{rows} grid for:", ", ".join(param_specs))

        for (ax, field) in zip_longest(axes.flatten(), param_specs):
            if field:
                bins, label = param_specs[field]
                data = [row.get(field, None) for row in read_param_sets_from_csvs(csvs)]
                if verbose:
                    print(f"Plotting histogram for {len(data):,} {field} values.")
                ax.hist(data, bins=bins)
                ax.set_xlabel(label)
                ax.tick_params(axis="both", which="both", direction="in",
                               top=True, bottom=True, left=True, right=True)
                ax.set_yscale(yscale)
            else:
                ax.axis("off") # remove the unused ax

        if verbose:
            print("Saving histogram plot to", plot_file)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file, dpi=100) # dpi is ignored for vector formats
        plt.close()


def plot_formal_test_dataset_hr_diagram(targets_cfg: Dict[str, any],
                                        verbose: bool=True):
    """
    Plots a log(L) vs log(Teff) H-R diagram with ZAMS line. Returns the figure
    of the plot and it is up to calling code to show or save this.

    :targets_cfg: the config data to plot from
    :verbose: whether to print out progress messages
    :returns: the Figure
    """
    if verbose:
        print("Plotting log(Teff) vs log(L) 'H-R' diagram")
    if verbose:
        print("Loading MIST isochrone for ZAMS data")

    fig = plt.figure(figsize=(6, 4), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    for comp, fillstyle in [("A", "full"), ("B", "none") ]:
        # Don't bother with error bars as this is just an indicative distribution.
        x = np.log10([cfg.get(f"Teff{comp}", None) or 0 for _, cfg in targets_cfg.items()])
        y = [cfg.get(f"logL{comp}", None) or 0 for _, cfg in targets_cfg.items()]

        ax.errorbar(x, y, fmt = "o", fillstyle = fillstyle, linewidth = 0.5,
                    ms = 7., markeredgewidth=0.5, c='tab:blue', label=f"Star{comp}")

        if verbose:
            print(f"Star {comp}: log(x) range [{min(x):.3f}, {max(x):.3f}],",
                               f"log(y) range [{min(y):.3f}, {max(y):.3f}]")

    # Now plot a ZAMS line from the MIST on the same criteria
    mist_isos = MistIsochrones(metallicities=[0.0])
    zams = mist_isos.lookup_zams_params(feh=0.0, cols=["log_Teff", "log_L"])
    ax.plot(zams[0], zams[1], c="k", ls=(0, (15, 5)), linewidth=0.5, label="ZAMS", zorder=-10)

    format_axes(ax, xlim=(4.45, 3.35), ylim=(-2.6, 4.5),
                xlabel= r"$\log{(\mathrm{T_{eff}\,/\,K})}$",
                ylabel=r"$\log{(\mathrm{L\,/\,L_{\odot}})}$")
    return fig


def plot_dataset_instance_mags_features(dataset_files: Iterable[Path],
                                        chosen_targets: List[str]=None,
                                        mags_bins: int=4096,
                                        mags_wrap_phase: float=0.75,
                                        cols: int=3,
                                        **format_kwargs) -> Figure:
    """
    Utility function to produce a plot of the requested dataset instance's mags feature

    :dataset_files: the set of dataset files to parse
    :identifier: the identifier of the instance
    :output: where to send the plot. Either a Path to save to or an existing axes
    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the figure
    """
    # pylint: disable=too-many-locals
    # Get the instances for each matching target
    instances = [*deb_example.iterate_dataset(dataset_files, mags_bins, mags_wrap_phase,
                                              identifiers=chosen_targets)]

    rows = math.ceil(len(instances) / cols)
    fig, axes = plt.subplots(rows, cols, sharex="all", sharey="all",
                             figsize=(cols*4, rows*3), tight_layout=True)

    # Infer the common phase data from the details of the bins we're given
    phases = np.linspace(mags_wrap_phase-1, mags_wrap_phase, mags_bins)

    phase_start = round(min(phases), 1)
    minor_xticks = np.arange(phase_start, phase_start + 1.0 if phase_start < 0.0 else 1.1, 0.1)
    major_xticks = [0.0, 0.5]
    ymin, ymax = 0, 0
    for ix, (ax, instance) in enumerate(zip_longest(axes.flatten(), instances)):
        if instance:
            # Only the mags are stored in the dataset. Infer the x/phase data
            (identifier, model_feature, _, _) = instance
            ax.scatter(x=phases, y=model_feature, marker=".", s=0.25, label=identifier)

            # Find the y-range which covers all the instances
            ylim = ax.get_ylim()
            ymin = min(ymin, min(ylim)) # pylint: disable=nested-min-max
            ymax = max(ymax, max(ylim)) # pylint: disable=nested-min-max

            ax.set_xticks(major_xticks, minor=False)
            ax.set_xticklabels(major_xticks)
            ax.set_xticks(minor_xticks, minor=True)

            # Work out whether we need a label on each axis. Basically only down the left
            # and at the bottom, if no further axes are going to appear below this one.
            ylabel = "Relative magnitude (mag)" if ix % cols == 0 else None
            xlabel = "Phase" if ix >= len(instances)-cols else None

            # We'll rely on the caller to config the output if it's an Axes
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, legend_loc="best", **format_kwargs)
        else:
            ax.axis("off") # remove the unused ax

    # Now we have the maximum extent of the mags go back through setting the ylims and phase vlines
    for ix, ax in enumerate(axes.flatten()):
        if ix < len(instances):
            ax.set_ylim((ymax, ymin)) # Has side effect of inverting the y-axis
            ax.vlines(major_xticks, ymin, ymax, ls="--", color="k", lw=.5, alpha=.25, zorder=-10)

    return fig
