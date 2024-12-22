""" Training and testing specific plots. """
from typing import List, Dict, Iterable
import math
from pathlib import Path
from itertools import zip_longest

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import binned_statistic
from uncertainties import UFloat, unumpy

from .datasets import iterate_dataset
from .plotting import format_axes
from .param_sets import read_from_csvs, get_field_names_from_csvs
from .mistisochrones import MistIsochrones

all_param_captions = {
    "rA_plus_rB":   r"$r_{\rm A}+r_{\rm B}$",
    "k":            r"$k$",
    "inc":          r"$i$",
    "J":            r"$J$",
    "qphot":        r"$q_{phot}$",
    "ecosw":        r"$e\,\cos{\omega}$",
    "esinw":        r"$e\,\sin{\omega}$",
    "L3":           r"$L_{\rm 3}$",
    "bP":           r"$b_{\rm P}$",
    "ecc":          r"$e$",
    "e":            r"$e$",
}

# The full set of parameters available for histograms, their #bins and plot labels
all_histogram_params = {
    "rA_plus_rB":   (100, r"$r_{\rm A}+r_{\rm B}$"),
    "k":            (100, r"$k$"),
    "inc":          (100, r"$i~(^{\circ})$"),
    "J":            (100, r"$J$"),
    "qphot":        (100, r"$q_{\rm phot}$"),
    "ecosw":        (100, r"$e\,\cos{\omega}$"),
    "esinw":        (100, r"$e\,\sin{\omega}$"),
    #"L3":           (100, r"$L_3$"), # currently always zero
    "bP":           (100, r"$b_{\rm P}$"),
    "bS":           (100, r"$b_{\rm S}$"),
    "ecc":          (100, r"$e$"),
    "omega":        (100, r"$\omega~(^{\circ})$"),
    "rA":           (100, r"$r_{\rm A}$"),
    "rB":           (100, r"$r_{\rm B}$"),
    "phiS":         (100, r"$\phi_{\rm S}$"),
    "dS_over_dP":   (100, r"$d_{\rm S}/d_{\rm P}$"),
    "RA":           (100, r"$R_{\rm A}~(\text{R}_{\odot})$"),
    "RB":           (100, r"$R_{\rm B}~(\text{R}_{\odot})$"),
    "MA":           (100, r"$M_{\rm A}~(\text{M}_{\odot})$"),
    "MB":           (100, r"$M_{\rm B}~(\text{M}_{\odot})$"),
    "snr":          (100, r"$S/N$"),
    "apparent_mag": (100, r"apparent magnitude (mag)")
}

# Standard width & heights for 6:4, 6:5 & square aspect ratios where plots will generally be either
# 2 (half page) or 4 (whole page) COL_WIDTH wide, as many of the plots are grids of axes.
# Width set so that font size ~matches when 2x2 plot is fitted to one column of two col LaTeX doc.
COL_WIDTH = 2.7
ROW_HEIGHT_6_4 = COL_WIDTH * 2/3
ROW_HEIGHT_6_5 = COL_WIDTH * 5/6
ROW_HEIGHT_SQUARE = COL_WIDTH

def plot_dataset_histograms(csv_files: Iterable[Path],
                            params: List[str]=None,
                            cols: int=3,
                            yscale: str="log",
                            verbose: bool=True):
    """
    Saves histogram plots to a single figure on a grid of axes. The params will be plotted
    in the order they are listed, scanning from left to right and down. These are generated
    from the dataset's CSV files as they may plot params not written to the dataset tfrecords.

    :csv_files: a list of the dataset's csv files
    :params: the list of parameters to plot (in this order), or those in all_histogram_params
    if None. In either case params will only be plotted if they are present in the csv files.
    :cols: the width of the axes grid (the rows automatically adjust)
    :yscale: set to "linear" or "log" to control the y-axis scale
    :verbose: whether to print verbose progress/diagnostic messages
    """
    # pylint: disable=too-many-arguments, too-many-locals
    csv_files = sorted(csv_files)   # Happy for this to error if there's a problem
    csv_params = get_field_names_from_csvs(csv_files)
    if params:
        plot_params = [p for p in params if p in all_histogram_params]
    else:
        plot_params = list(all_histogram_params.keys())
    param_specs = { p: all_histogram_params[p] for p in plot_params if p in csv_params }

    fig = None
    if param_specs and csv_files:
        rows = math.ceil(len(param_specs) / cols)
        fig, axes = plt.subplots(rows, cols, sharey="all", constrained_layout=True,
                                 figsize=(cols * COL_WIDTH, rows * ROW_HEIGHT_6_5))
        if verbose:
            print(f"Plotting histograms in a {cols}x{rows} grid for:", ", ".join(param_specs))

        for (ax, field) in zip_longest(axes.flatten(), param_specs):
            if field:
                bins, label = param_specs[field]
                data = [row.get(field, None) for row in read_from_csvs(csv_files)]
                if verbose:
                    print(f"Plotting histogram for {len(data):,} {field} values.")
                ax.hist(data, bins=bins)
                ax.set_xlabel(label)
                ax.tick_params(axis="both", which="both", direction="in",
                               top=True, bottom=True, left=True, right=True)
                ax.set_yscale(yscale)
            else:
                ax.axis("off") # remove the unused ax
    return fig


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

    fig = plt.figure(figsize=(2 * COL_WIDTH, 2 * ROW_HEIGHT_6_4), constrained_layout=True)
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
    if verbose:
        print("Loading MIST isochrone for ZAMS data")
    mist_isos = MistIsochrones(metallicities=[0.0])
    zams = mist_isos.lookup_zams_params(feh=0.0, cols=["log_Teff", "log_L"])
    ax.plot(zams[0], zams[1], c="k", ls=(0, (15, 5)), linewidth=0.5, label="ZAMS", zorder=-10)

    format_axes(ax, xlim=(4.45, 3.35), ylim=(-2.6, 4.5),
                xlabel= r"$\log{(T_{\rm eff}\,/\,{\rm K})}$",
                ylabel=r"$\log{(L\,/\,{\rm L_{\odot}})}$")
    return fig


def plot_dataset_instance_mags_features(dataset_files: Iterable[Path],
                                        chosen_targets: List[str]=None,
                                        mags_bins: int=4096,
                                        mags_wrap_phase: float=0.75,
                                        cols: int=3,
                                        max_instances: int=np.inf,
                                        **format_kwargs) -> Figure:
    """
    Utility function to produce a plot of the requested dataset instance's mags feature

    :dataset_files: the set of dataset files to parse
    :chosen_targets: a list of ids to find and plot, or all as yielded (up to max_instances) if None
    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :max_instances: the maximum number of instances to plot
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the figure
    """
    # pylint: disable=too-many-locals
    # Get the instances for each matching target
    instances = [*iterate_dataset(dataset_files, mags_bins, mags_wrap_phase,
                                  identifiers=chosen_targets, max_instances=max_instances)]

    rows = math.ceil(len(instances) / cols)
    fig, axes = plt.subplots(rows, cols, sharex="all", sharey="all", constrained_layout=True,
                             figsize=(cols * COL_WIDTH, rows * ROW_HEIGHT_6_5))

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

            # We'll rely on the caller to config the output if it's an Axes
            format_axes(ax, legend_loc="lower right", **format_kwargs)
        else:
            ax.axis("off") # remove the unused ax

    # Now we have the maximum extent of the mags go back through setting the ylims and phase vlines
    ymax += 0.1 # extend the y-axis so there is always space for the legend
    for ix, ax in enumerate(axes.flatten()):
        if ix < len(instances):
            ax.set_ylim((ymax, ymin)) # Has side effect of inverting the y-axis
            ax.vlines(major_xticks, ymin, ymax, ls="--", color="lightgray", lw=.5, zorder=-10)

    # Common x- and y-axis labels
    fig.supxlabel("Orbital Phase")
    fig.supylabel("Differential magnitude (mag)")
    return fig


def plot_limb_darkening_coeffs(lookup_table: np.ndarray[float],
                               coeffs_used: Iterable[Dict[str, float]] = None,
                               x_col: str="a",
                               y_col: str="b",
                               logg_col: str="logg",
                               teff_col: str="Teff",
                               **format_kwargs) -> Figure:
    """
    Will create a plot figure with a single set of axes upon which will be plotted
    the progression of the limb darkening coefficients (as indicated by x_col & y_col)
    for each distinct value of logg in the passed lookup table. On this will then be
    plotted any chosen (LDA1, LDA2) coefficient values.

    :lookup_table: the table of coefficients to compare the chosen with
    :coeffs_used: the chosen coeffs to plot over the lookups
    :x_col: the column of the lookup table to plot on the x-axis
    :y_col: the column of the lookup table to plot on the y-axis
    :logg_col: the logg column name; used to iterate over the distinct values
    :teff_col: the T_eff column name; used to sort the rows for each distinct logg
    :format_kwargs: kwargs to be passed on to format_axes()
    """
    fig, ax = plt.subplots(figsize=(2 * COL_WIDTH, 2 * ROW_HEIGHT_6_5), constrained_layout=True)

    # Plot the lookup data, joined up points of the coeffs for each distinct logg
    for logg in sorted(np.unique(lookup_table[logg_col])):
        params = np.sort(lookup_table[lookup_table[logg_col] == logg], order=teff_col)
        ax.plot(params[x_col], params[y_col], "-", lw=0.5, alpha=1/3)

        # Overplot with dot markers which scale with the T_eff as a proxy for M*
        size = params[teff_col] / 500
        ax.scatter(params[x_col], params[y_col], s=size, marker="o", alpha=1/2, zorder=5,
                   label=f"$\\log{{g}}={logg}$")

    # Overplot with the coeffs used as black cross(es)
    if isinstance(coeffs_used, Dict) or coeffs_used is None:
        coeffs_used = [coeffs_used]
    for i, used in enumerate(coeffs_used):
        cf = (used["LDA1"], used["LDA2"])
        ax.scatter(cf[0], cf[1], marker="+", c="k", s=150, zorder=10, label=f"used[{i}]\n{cf}")

    if format_kwargs is None:
        format_kwargs = {}

    # If not given, extend the limits so that there is room for the potentially rather large legend
    if format_kwargs.get("legend_loc", None):
        (xfrom, xto) = ax.get_xlim()
        format_kwargs.setdefault("xlim", (xfrom-0.01, xto+0.15))
        (yfrom, yto) = ax.get_ylim()
        format_kwargs.setdefault("ylim", (yfrom-0.05, yto+0.05))

    format_axes(ax, **format_kwargs)

    ax.text(ax.get_xlim()[0] + 0.02, ax.get_ylim()[0] + 0.02, # data coords
            r"Larger markers indicate higher $T_{\rm eff}$") 
    return fig


def plot_predictions_vs_labels(predictions: np.ndarray[UFloat],
                               labels: np.ndarray[UFloat],
                               transit_flags: np.ndarray[bool],
                               selected_params: List[str]=None,
                               show_errorbars: bool=None,
                               xlabel_prefix: str="label",
                               ylabel_prefix: str="predicted") -> Figure:
    """
    Will create a plot figure with a grid of axes, one per label, showing the
    predictions vs label values. It is up to calling code to show or save the figure.

    :predictions: the prediction values
    :labels: the label values
    :transit_flags: the associated transit flags; points where the transit flag is True
    are plotted as a filled shape otherwise as an empty shape
    :selected_params: a subset of the full list of prediction/label params to render
    :show_errorbars: whether to plot errorbars for predictions and labels - if not set the function
    will plot errorbars if there are non-zero error/sigma values in the predictions
    :xlabel_prefix: the prefix text for the labels/x-axis label
    :ylabel_prefix: the prefix text for the predictions/y-axis label
    :returns: the Figure
    """
    # pylint: disable=too-many-arguments, too-many-locals

    # We plot the params common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want requested names or the labels to set order
    if selected_params is None:
        selected_params = [n for n in labels.dtype.names if n in predictions.dtype.names]
    elif isinstance(selected_params, str):
        selected_params = [selected_params]
    params = { n: all_param_captions[n] for n in selected_params }

    # Special aspect ratio for each axes of 3.0:2.9, slightly wider than high, to look balanced
    # with a sqaure plot area and slightly more width for y-tick labels than those for x-ticks.
    cols = 2
    rows = math.ceil(len(params) / cols)
    fig, axes = plt.subplots(rows, cols, constrained_layout=True,
                             figsize=(cols * COL_WIDTH, rows * ROW_HEIGHT_SQUARE))
    axes = axes.flatten()

    if transit_flags is None:
        transit_flags = np.zeros((labels.shape[0]), dtype=bool)

    print(f"Plotting scatter plot {rows}x{cols} grid for: {', '.join(params.keys())}")
    for (ax, param_name) in zip_longest(axes.flatten(), params.keys()):
        if param_name:
            lbl_vals = unumpy.nominal_values(labels[param_name])
            lbl_sigmas = unumpy.std_devs(labels[param_name])
            pred_vals = unumpy.nominal_values(predictions[param_name])
            pred_sigmas = unumpy.std_devs(predictions[param_name])

            # Plot a diagonal line for exact match
            dmin, dmax = min(lbl_vals.min(), pred_vals.min()), max(lbl_vals.max(), pred_vals.max()) # pylint: disable=nested-min-max
            if param_name in ["rA_plus_rB", "k", "J", "bP"]:
                dmin = min(0, dmin)
            drange = dmax - dmin
            dmore = 0.125 * drange
            diag = (dmin - dmore, dmax + dmore)
            ax.plot(diag, diag, color="gray", linestyle="-", linewidth=0.5)

            # If we have lots of data, reduce the size of the marker and add in an alpha
            (fmt, ms, alpha) = ("o", 5.0, 1.0) if len(lbl_vals) < 100 else (".", 2.0, 0.25)

            # Plot the preds vs labels, with those with transits filled.
            if show_errorbars is None:
                show_errorbars = max(np.abs(pred_sigmas)) > 0
            for tmask, transiting in [(transit_flags, True), (~transit_flags, False)]:
                if any(tmask):
                    (f, z) = ("full", 10) if transiting else ("none", 0)
                    if show_errorbars:
                        ax.errorbar(x=lbl_vals[tmask], y=pred_vals[tmask],
                                    xerr=lbl_sigmas[tmask], yerr=pred_sigmas[tmask],
                                    fmt=fmt, c="tab:blue", ms=ms, lw=1.0, alpha=alpha,
                                    capsize=2.0, markeredgewidth=0.5, fillstyle=f, zorder=z)
                    else:
                        ax.errorbar(x=lbl_vals[tmask], y=pred_vals[tmask], fmt=fmt, c="tab:blue",
                                    alpha=alpha, ms=ms, lw=1.0, fillstyle=f, zorder=z)

            param_caption = params[param_name]
            format_axes(ax, xlim=diag, ylim=diag, xlabel=f"{xlabel_prefix} {param_caption}",
                        ylabel=f"{ylabel_prefix} {param_caption}")

            # Make sure the plot areas are squared and have similar label areas.
            ax.set_aspect("equal", "box")
            ax.tick_params("y", rotation=90)

            # We want up to 5 tick labels at suitable points across the range of values.
            if param_name == "inc":
                maj_ticks = np.arange(50, 90.1, 5 if drange < 25 else 10)
            elif param_name in ["ecosw", "esinw"]:
                maj_ticks = [-0.4, -0.2, 0, 0.2, 0.4] if drange < 1 else [-0.8, -0.4, 0.0, 0.4, 0.8]
            else:
                lim = max(diag)
                for tick_step in [0.1, 0.2, 0.5, 1, 2, 2.5, 5, 10]:
                    if lim / tick_step < 5:
                        break
                maj_ticks = np.arange(0, lim, tick_step)
            maj_ticks = [t for t in maj_ticks if diag[0] < t < diag[1]]
            ax.set_yticks(maj_ticks, minor=False)
            ax.set_xticks(maj_ticks, minor=False)
        else:
            ax.axis("off") # remove the unused ax
    return fig


def plot_binned_mae_vs_labels(residuals: np.rec.recarray[UFloat],
                              labels: np.rec.recarray[UFloat],
                              selected_params: List[str]=None,
                              num_bins: float=100,
                              indicate_bin_counts: bool=False,
                              xlabel: str="label value",
                              ylabel: str="mean absolute error",
                              **format_kwargs) -> Figure:
    """
    Will create a plot figure with a single set of axes, with the MAE vs label values
    for one or more labels broken down into equal sized bins. It is intended to show
    how the prediction accuracy varies over the range of the labels.

    :residuals: the residual values
    :labels: the label values
    :selected_params: a subset of the full list of prediction/label params to render
    :num_bins: the number of equal sized bins to apply to the data
    :indicate_bin_counts: give an indication of each bin's count by its marker size
    :xlabel: the label to give the x-axis
    :ylabel: the label to give the y-axis
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the figure
    """
    # pylint: disable=too-many-arguments, too-many-locals
    if selected_params is None:
        selected_params = [n for n in labels.dtype.names if n in residuals.dtype.names]
    elif isinstance(selected_params, str):
        selected_params = [selected_params]
    params = { n: all_param_captions[n] for n in selected_params }

    print("Plotting binned MAE vs label values for:", ", ".join(params))
    fig, ax = plt.subplots(figsize=(2 * COL_WIDTH, 2 * ROW_HEIGHT_6_4), constrained_layout=True)

    # We need to know the extent of all the data beforehand, so we can apply equivalent bins to all
    lbl_vals = unumpy.nominal_values(labels[selected_params].tolist())
    bins = np.linspace(lbl_vals.min(), lbl_vals.max(), num_bins+1)
    for ix, param_name in enumerate(params):
        abs_resids = np.abs(unumpy.nominal_values(residuals[param_name]))

        means, bin_edges, _ = binned_statistic(lbl_vals[:, ix], abs_resids, "mean", bins)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centres = bin_edges[1:] - bin_width / 2

        if indicate_bin_counts:
            counts, _, _ = binned_statistic(lbl_vals[:, ix], abs_resids, "count", bins)
            marker_sizes = 1.0 * (counts/10)
            alpha = 0.5 # more likely to overlap
        else:
            marker_sizes = 5.0
            alpha = 0.75

        ax.scatter(bin_centres, means, s=marker_sizes, alpha=alpha, label=params[param_name])

    format_axes(ax, xlabel=xlabel, ylabel=ylabel, legend_loc="best", **format_kwargs)
    return fig


def plot_prediction_boxplot(predictions: np.rec.recarray[UFloat],
                            show_zero_value_line: bool=True,
                            show_fliers: bool=False,
                            **format_kwargs):
    """
    Plot a box plot of the prediction distribution (the last axis of predictions).

    The box plot is set up to focus on the distribution about the median value, with the meadian
    shown with a line and a box bounding the (2nd & 3rd) interquartile range, the whiskers covering
    values within x1.5 interquartile range from the box, and fliers/outliers (if enabled) beyond.

    :predictions: the predictions to plot; recarray[UFloat] of shape (#instances, #labels)
    :show_zero_value_line: whether to draw a horizontal line at zero
    :show_fliers: if true, outliers are plotted beyond the box_plot whiskers
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the figure
    """
    # We're only interested in the nominals. Get this into a format matplotlib can handle
    xdata = unumpy.nominal_values(predictions.tolist())

    # For customizations https://matplotlib.org/stable/gallery/statistics/boxplot.html
    fig, ax = plt.subplots(figsize=(2 * COL_WIDTH, 2 * ROW_HEIGHT_6_4), constrained_layout=True)
    flier_props = { "marker": "x", "alpha": 0.5 }
    ax.boxplot(xdata, showmeans=False, meanline=True, vert=True, patch_artist=False,
               showfliers=show_fliers, flierprops=flier_props)

    pub_labels = [all_param_captions[k] for k in predictions.dtype.names]
    ax.set_xticks(ticks=[r+1 for r in range(len(pub_labels))], labels=pub_labels)

    if show_zero_value_line:
        (xmin, xmax) = ax.get_xlim()
        ax.hlines([0.0], xmin, xmax, linestyles="--", color="k", lw=.5, alpha=.5, zorder=-10)

    if format_kwargs:
        format_axes(ax, **format_kwargs)

    # Hide minor x-ticks as they have no meaning in this context
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    return fig
