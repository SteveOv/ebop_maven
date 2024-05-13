""" Matplotlib plotting helper functions. """
from typing import Tuple, List, Dict, Union
import math
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

import model_testing

def format_axes(ax: Axes, title: str=None,
                xlabel: str=None, ylabel: str=None,
                xticklable_top: bool=False, yticklabel_right: bool=False,
                invertx: bool=False, inverty: bool=False,
                xlim: Tuple[float, float]=None, ylim: Tuple[float, float]=None,
                minor_ticks: bool=True, legend_loc: str=None):
    """
    General purpose formatting function for a set of Axes. Will carry out the
    formatting instructions indicated by the arguments and will set all ticks
    to internal and on all axes.

    :ax: the Axes to format
    :title: optional title to give the axes, overriding prior code - set to "" to surpress
    :xlabel: optional x-axis label text to set, overriding prior code - set to "" to surpress
    :ylabel: optional y-axis label text to set, overriding prior code - set to "" to surpress
    :xticklabel_top: move the x-axis ticklabels and label to the top
    :yticklabel_right: move the y-axis ticklabels and label to the right
    :invertx: invert the x-axis
    :inverty: invert the y-axis
    :xlim: set the lower and upper limits on the x-axis
    :ylim: set the lower and upper limits on the y-axis
    :minor_ticks: enable or disable minor ticks on both axes
    :legend_loc: if set will enable to legend and set its position.
    For available values see matplotlib legend(loc="")
    """
    # pylint: disable=too-many-arguments
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if invertx:
        ax.invert_xaxis()
    if inverty:
        ax.invert_yaxis()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if minor_ticks:
        ax.minorticks_on()
    else:
        ax.minorticks_off()
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, bottom=True, left=True, right=True)
    if xticklable_top:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    if yticklabel_right:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    if legend_loc:
        ax.legend(loc=legend_loc)


def plot_predictions_vs_labels(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        transit_flags: List[bool],
        selected_labels: List[str]=None,
        show_errorbars: bool=None,
        reverse_scaling: bool=False) -> Figure:
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
    :returns: the Figure
    """
    # pylint: disable=too-many-arguments, too-many-locals
    all_pub_labels = {
        "rA_plus_rB": "$r_A+r_B$",
        "k": "$k$",
        "inc": "$i$",
        "bP": "$b_P$",
        "J": "$J$",
        "ecosw": r"$e\cos{\omega}$",
        "esinw": r"$e\sin{\omega}$",
        "L3": "$L_3$",
    }

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
                    xlabel=f"label {ax_label}", ylabel=f"predicted {ax_label}")

        # Make sure the plots are squared and have the same ticks
        ax.set_aspect("equal", "box")
        ax.set_yticks([t for t in ax.get_xticks() if diag[0] < t < diag[1]])
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
    isos = _read_mist_isos()

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
    x = [eep[eep["phase"] == 0][0]["log_Teff"] for eep in isos]
    y = [eep[eep["phase"] == 0][0]["log_L"] for eep in isos]
    ax.plot(x, y, c="k", ls=(0, (15, 5)), linewidth=0.5, label="ZAMS", zorder=-10)
    format_axes(ax, xlim=(4.45, 3.35), ylim=(-2.6, 4.5),
                xlabel= r"$\log{(\mathrm{T_{eff}\,/\,K})}$",
                ylabel=r"$\log{(\mathrm{L\,/\,L_{\odot}})}$")
    return fig


def _read_mist_isos(file_name: Path=None) -> List:
    """
    Read in a MIST iso file and return an isos list.
    Just a cut down copy of the sample MIST code.
    """
    if not file_name:
        file_name = Path.cwd() / "config/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso"

    isos = []
    with open(file_name, mode="r", encoding="utf8") as isof:
        content = [line.split() for line in isof]

        #read one block for each isochrone
        counter = 0
        data = content[8:]
        num_ages = int(content[6][-1])
        for _ in range(num_ages):
            # grab info for each isochrone
            num_eeps = int(data[counter][-2])
            num_cols = int(data[counter][-1])
            hdr_list = data[counter+2][1:]
            formats = tuple([np.int32]+[np.float64 for i in range(num_cols-1)])
            iso = np.zeros((num_eeps),{'names':tuple(hdr_list),'formats':tuple(formats)})

            # read through EEPs for each isochrone
            for eep in range(num_eeps):
                iso_chunk = data[3+counter+eep]
                iso[eep]=tuple(iso_chunk)
            isos.append(iso)
            counter+=3+num_eeps+2
    return isos


if __name__ == "__main__":

    skip = ["V402 Lac", "V456 Cyg"] # Neither are suitable for JKTEBOP fitting
    with open(Path.cwd() / "config/formal-test-dataset.json", mode="r", encoding="utf8") as f:
        targets_config = { t_name: v for (t_name, v) in json.load(f).items() if t_name not in skip }

    plot_formal_test_dataset_hr_diagram(targets_config, True).savefig(
                                                    Path.cwd() / "drop/plots/hr-logl-vs-teff.eps")
