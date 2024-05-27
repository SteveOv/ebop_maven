""" Support for common plots and formatting. """
# pylint: disable=too-many-arguments
from typing import Tuple, List, Union

from matplotlib.axes import Axes
import numpy as np

from lightkurve import LightCurve, FoldedLightCurve
from astropy.time import Time

label_text = {
    "rA_plus_rB": "$r_A+r_B$",
    "k": "$k$",
    "inc": "$i$",
    "J": "$J$",
    "ecosw": r"$e\,\cos{\omega}$",
    "esinw": r"$e\,\sin{\omega}$",
    "L3": "$L_3$",
    "bP": "$b_P$",
}

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
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()


def plot_lightcurve_on_axes(ax: Axes,
                            lc: LightCurve,
                            ydata_column: str="delta_mag",
                            ydata_label: str=None,
                            primary_epoch: Time=None,
                            primary_epoch_label: str=None,
                            **format_kwargs):
    """
    Plot the passed LightCurve on the passed Axes and optionally overlay a marker
    indicating the position of the primary epoch.

    :ax: the Axes to plot to
    :lc: the LightCurve to plot
    :ydata_column: the name of the column to plot on the y-axis
    :ydata_label: the label, if any, to assign to the plotted data points
    :primary_epoch: optional primary epoch time to indicate
    :format_kwargs: kwargs to be passed on to format_axes()
    """
    # Plot the light curve itself
    lc.scatter(ax=ax, column=ydata_column, s=5, alpha=0.5, label=ydata_label)

    # The optional Primary Epoch marker
    if primary_epoch is not None:
        pe_value = primary_epoch.value
        nearest_ix = np.abs(lc.time.value - pe_value).argmin()
        pe_flux = lc[ydata_column][nearest_ix].value
        ax.scatter([pe_value], [pe_flux], marker="x", s=50., lw=1., c="tab:red",
                   label=primary_epoch_label, zorder=10)

    if format_kwargs:
        format_axes(ax, **format_kwargs)


def plot_folded_lightcurve_on_axes(ax: Axes,
                                flc: FoldedLightCurve,
                                ydata_column: str="delta_mag",
                                ydata_label: str=None,
                                show_phase_vlines: bool=False,
                                overlay_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]=None,
                                overlay_data_label: str=None,
                                **format_kwargs):
    """
    Plot the passed FoldedLightCurve on the passed Axes and optionally overlay
    another set of fitted flux data.

    :ax: the Axes to plot to
    :lc: the LightCurve to plot
    :ydata_column: the name of the column to plot on the y-axis
    :ydata_label: the label, if any, to assign to the plotted data points
    :show_phase_vlines: whether to show optional vertical lines a phases 0 and 0.5
    :overlay_data: optional phase folded data to overlay on the light curve
    :overlay_data_label: optional label for the overlay data
    :format_kwargs: kwargs to be passed on to format_axes()
    """
    flc.scatter(ax=ax, column=ydata_column, s=20, alpha=.25, label=ydata_label)

    if show_phase_vlines:
        (ymin, ymax) = ax.get_ylim()
        ax.vlines([0.0, 0.5], ymin, ymax, linestyles="--", color="k", lw=.5, alpha=.5, zorder=-10)

    if overlay_data is not None:
        if isinstance(overlay_data, tuple) and len(overlay_data) >= 2:
            (xdata, ydata) = overlay_data
        elif isinstance(overlay_data, np.ndarray) and overlay_data.shape[0] >= 2:
            (xdata, ydata) = overlay_data[0], overlay_data[1]
        else:
            raise ValueError("Unsure of overlay_data format. " \
                             +"Expect tuple of ndarrays as (xdata, ydata) or ndarray[xdata, ydata]")
        ax.scatter(xdata, ydata, zorder=10,
                   c="k", alpha=.75, marker=".", s=5, lw=1, label=overlay_data_label)

    if format_kwargs:
        format_axes(ax, **format_kwargs)


def plot_prediction_distributions_on_axes(ax: Axes,
                                          predictions: np.ndarray,
                                          label_names: List[str],
                                          violin_plot: bool=False,
                                          plot_zero_value_line: bool=True,
                                          **format_kwargs):
    """
    Plot a violin or box plot of the prediction distribution (the last axis of predictions).

    :ax: the Axes to plot to
    :predictions: the predictions to plot; ndarray of shape (#labels, #iterations)
    :label_names: the names of each label on predictions[0]
    :violin_plot: whether to plot a violin plot (True) or a box plot (False)
    :plot_zero_value_line: whether to draw a horizontal line at zero
    :format_kwargs: kwargs to be passed on to format_axes()
    """
    # We can only handle 1 instance so take the first if we have full (#insts, #labels, #iterations)
    xdata = predictions[0] if len(predictions.shape) == 3 else predictions

    # Get the predictions in the right shape to plot along the label axis.
    if xdata.shape[0] == len(label_names):
        xdata = xdata.transpose()

    if violin_plot:
        ax.violinplot(xdata, showmeans=True, vert=True)
    else:
        ax.boxplot(xdata, showmeans=True, vert=True, patch_artist=True)

    label_names = [label_text[k] or k for k in label_names]
    ax.set_xticks(ticks=[r+1 for r in range(len(label_names))], labels=label_names)

    if plot_zero_value_line:
        (xmin, xmax) = ax.get_xlim()
        ax.hlines([0.0], xmin, xmax, linestyles="--", color="k", lw=.5, alpha=.5, zorder=-10)

    if format_kwargs:
        format_axes(ax, **format_kwargs)

    # Hide minor x-ticks as they have no meaning in this context
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
