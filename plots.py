""" Matplotlib plotting helper functions.  *** Funcs to be move away *** """
from typing import Tuple, List, Dict, Union
import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import binned_statistic

from ebop_maven.plotting import format_axes
from ebop_maven.libs.mistisochrones import MistIsochrones
import model_testing

all_pub_labels = {
    "rA_plus_rB":   "$r_A+r_B$",
    "k":            "$k$",
    "inc":          "$i$",
    "J":            "$J$",
    "ecosw":        r"$e\cos{\omega}$",
    "esinw":        r"$e\sin{\omega}$",
    "L3":           "$L_3$",
    "bP":           "$b_P$",
}

def plot_predictions_vs_labels(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        transit_flags: List[bool],
        selected_labels: List[str]=None,
        show_errorbars: bool=None,
        reverse_scaling: bool=False,
        xlabel_prefix: str="label",
        ylabel_prefix: str="predicted") -> Figure:
    """
    Will create a plot figure with a grid of axes, one per label, showing the
    predictions vs label values. It is up to calling code to show or save the figure.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :transit_flags: the associated transit flags; points where the transit flag is True
    are plotted as a filled shape otherwise as an empty shape
    :selected_labels: a subset of the full list of labels/prediction names to render
    :show_errorbars: whether to plot errorbars - if not set the function will plot errorbars
    if there are non-zero error/sigma values in the predictions
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    :xlabel_prefix: the prefix text for the labels/x-axis label
    :ylabel_prefix: the prefix text for the predictions/y-axis label
    :returns: the Figure
    """
    # pylint: disable=too-many-arguments, too-many-locals

    # We plot the keys common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want names or the labels to set order
    if selected_labels is None:
        selected_labels = list(all_pub_labels.keys())
    pub_labels = { k: all_pub_labels[k] for k in selected_labels if k in predictions[0].keys() }

    cols = 2
    rows = math.ceil(len(pub_labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.9), constrained_layout=True)
    axes = axes.flatten()

    if transit_flags is None:
        transit_flags = [False] * len(labels)

    print(f"Plotting scatter plot {rows}x{cols} grid for: {', '.join(pub_labels.keys())}")
    for ax_ix, (lbl_name, ax_label) in enumerate(pub_labels.items()):
        (lbl_vals, pred_vals, pred_sigmas, _) = model_testing.get_label_and_prediction_raw_values(
                                                labels, predictions, [lbl_name], reverse_scaling)

        # Plot a diagonal line for exact match
        dmin, dmax = min(lbl_vals.min(), pred_vals.min()), max(lbl_vals.max(), pred_vals.max()) # pylint: disable=nested-min-max
        dmore = 0.1 * (dmax - dmin)
        diag = (dmin - dmore, dmax + dmore)
        ax = axes[ax_ix]
        ax.plot(diag, diag, color="gray", linestyle="-", linewidth=0.5)

        # Plot the preds vs labels, with those with transits highlighted
        # We want to set the fillstyle by transit flag which means plotting each item alone
        show_errorbars = show_errorbars if show_errorbars else max(np.abs(pred_sigmas)) > 0
        for x, y, yerr, transiting in zip(lbl_vals, pred_vals, pred_sigmas, transit_flags):
            (f, z) = ("full", 10) if transiting else ("none", 0)
            if show_errorbars:
                ax.errorbar(x=x, y=y, yerr=yerr, fmt="o", c="tab:blue", ms=5.0, lw=1.0,
                            capsize=2.0, markeredgewidth=0.5, fillstyle=f, zorder=z)
            else:
                ax.errorbar(x=x, y=y, fmt="o", c="tab:blue", ms=5.0, lw=1.0, fillstyle=f, zorder=z)

        format_axes(ax, xlim=diag, ylim=diag,
                    xlabel=f"{xlabel_prefix} {ax_label}", ylabel=f"{ylabel_prefix} {ax_label}")

        # Make sure the plots are squared and have the same ticks
        ax.set_aspect("equal", "box")
        ax.set_yticks([t for t in ax.get_xticks() if diag[0] < t < diag[1]])
    return fig


def plot_binned_mae_vs_labels(
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        labels: List[Dict[str, float]],
        selected_labels: List[str]=None,
        num_bins: float=100,
        indicate_bin_counts: bool=False,
        xlabel: str="label value",
        ylabel: str="mean absolute error",
        **format_kwargs) -> Figure:
    """
    Will create a plot figure with a single set of axes, with the MAE vs label values
    for one or more labels broken down into equal sized bins. It is intended to show
    how the predictions accuracy varies over the range of the labels.

    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :labels: the labels values as a dict of labels per instance
    :selected_labels: a subset of the full list of labels/prediction names to render
    :num_bins: the number of equal sized bins to apply to the data
    :indicate_bin_counts: give an indication of each bin's count by its marker size
    :xlabel: the label to give the x-axis
    :ylabel: the label to give the y-axis
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the figure
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We plot the keys common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want input names or the labels to set order
    if selected_labels is None:
        selected_labels = list(all_pub_labels.keys())
    pub_labels = { k: all_pub_labels[k] for k in selected_labels if k in predictions[0].keys() }

    print("Plotting binned MAE vs label values for:", ", ".join(pub_labels.keys()))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    # We need to know the extent of the data beforehand, so we can apply equivalent bins to all
    (lbl_vals, pred_vals, _, resids) = model_testing.get_label_and_prediction_raw_values(
                                                    labels, predictions, list(pub_labels.keys()))
    resids = np.abs(resids)

    min_val = min(lbl_vals.min(), pred_vals.min())
    max_val = max(lbl_vals.max(), pred_vals.max())
    bins = np.linspace(min_val, max_val, num_bins+1)

    for ix, label_name in enumerate(pub_labels):
        means, bin_edges, _ = binned_statistic(lbl_vals[:, ix], resids[:, ix], "mean", bins)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centres = bin_edges[1:] - bin_width / 2

        if indicate_bin_counts:
            counts, _, _ = binned_statistic(lbl_vals[:, ix], resids[:, ix], "count", bins)
            marker_sizes = 1.0 * (counts/10)
            alpha = 0.5 # more likely to overlap
        else:
            marker_sizes = 5.0
            alpha = 0.75

        ax.scatter(bin_centres, means, s=marker_sizes, alpha=alpha, label=pub_labels[label_name])

    format_axes(ax, xlabel=xlabel, ylabel=ylabel, legend_loc="best", **format_kwargs)
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
