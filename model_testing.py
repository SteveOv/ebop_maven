"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
from typing import Union, List, Dict, Tuple
from io import TextIOBase
import sys
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model

from ebop_maven.libs import deb_example
from ebop_maven.estimator import Estimator
from ebop_maven import modelling

def test_model_against_formal_test_dataset(
        model: Union[Model, Path],
        test_dataset_dir: Path=Path("./datasets/formal-test-dataset"),
        test_dataset_config: Path=Path("./config/formal-test-dataset.json"),
        results_dir: Path=None,
        plot_results: bool=True):
    """
    Will test the indicated model file against the contents of the formal
    test dataset. The results for MC and non-MC estimates will be written
    to csv files in the indicated results directory. Will also need access
    to the config for the formal test dataset so that it can retrieve target
    names and flags.

    :model: the save model to test
    :test_dataset_dir: the location of the formal test dataset
    :test_dataset_config: Path to the config that created the test dataset
    :results_dir: the parent location to write the results csv file(s) or, if
    None, the /results/{model.name}/{trainset_name} subdirectory of the model location is used
    :plot_results: whether to produce a plot of the results vs labels
    """
    # Create our Estimator. It will tell us which labels it (& the model) can predict.
    estimator = Estimator(model)
    label_names = [*estimator.label_names_and_scales.keys()]

    print(f"\nLooking for the test dataset in '{test_dataset_dir}'.")
    tfrecord_files = list(test_dataset_dir.glob("**/*.tfrecord"))
    if len(tfrecord_files) == 0:
        raise IndexError("No dataset files found")

    # Don't iterate over the dataset; it's easier if we work with all the data (via a single batch)
    # As the lbls have come through a dataset pipeline any scaling will be applied.
    map_func = deb_example.create_map_func(labels=label_names)
    (ds_formal_test, inst_count) = deb_example.create_dataset_pipeline(tfrecord_files,
                                                                       10000, map_func)
    (test_mags, test_feats), label_vals = next(ds_formal_test.take(inst_count).as_numpy_iterator())

    # The features arrive in a tuple (array[shape(#inst, #bin, 1)], array[shape(#inst, #feat, 1)])
    # We need them in the input format for the Estimator [{"mags":.. , "feat1": ... }]
    fn = [*deb_example.extra_features_and_defaults.keys()]
    instance_features = [
        { "mags": tm, **{n:f[0] for (n,f) in zip(fn,tf)} } for tm, tf in zip(test_mags, test_feats)
    ]

    # Read additional target data from the config file
    target_names = [] * inst_count
    transit_flags = [False] * inst_count
    with open(test_dataset_config, mode="r", encoding="utf8") as f:
        targets = json.load(f)
        target_names = [f"{t}" for t in targets]
        transit_flags = [config.get("transits", False) for t, config in targets.items()]

    # Make the results directory
    if results_dir is None:
        parent = model.parent if isinstance(model, Path) else Path("./drop")
        results_dir = parent / f"results/{estimator.name}/formal-training-dataset"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run the predictions twice; with and without using MC Dropout enabled.
    # We get the labels into a [{"name": value}]*insts format for easier comparison with preds
    labels = [dict(zip(label_names, label_row)) for label_row in label_vals]
    for iterations, suffix in [(1, "nonmc"), (1000, "mc")]:
        # Make our prediction which will return [{"name": value, "name_sigma": sigma_value}]*insts
        # Here we don't want to unscale the predictions so we're consistent with evaluate().
        print(f"\nUsing an Estimator to make predictions on {inst_count} formal test instances.")
        predictions = estimator.predict(instance_features, iterations, unscale=False)

        results_file = results_dir / f"formal.test.{suffix}.csv"
        with open(results_file, mode="w", encoding="utf8") as of:
            predictions_vs_labels_to_csv(labels, predictions, estimator, target_names, to=of)

        print(f"\nSaved predictions and associated loss information to '{results_file}'")
        (_, _, _, ocs) = _get_label_and_prediction_raw_values(labels, predictions)
        print("--------------------------------")
        print(f"Total MAE ({suffix}): {np.mean(np.abs(ocs)):.9f}")
        print(f"Total MSE ({suffix}): {np.mean(np.power(ocs, 2)):.9f}")
        print("--------------------------------\n")

        with open(results_file.parent / f"{results_file.stem}.txt", "w", encoding="utf8") as of:
            table_of_predictions_vs_labels(labels, predictions, estimator, target_names, to=of)

        if plot_results:
            plot_file = results_file.parent / f"predictions_vs_labels_{suffix}.eps"
            plot_predictions_vs_labels(labels, predictions, transit_flags, plot_file)


def predictions_vs_labels_to_csv(
        labels: Dict[str, float],
        predictions: Union[Dict[str, float], Dict[str, Tuple[float, float]]],
        estimator: Estimator,
        row_headings: List[str]=None,
        selected_labels: List[str]=None,
        reverse_scaling: bool=False,
        to: TextIOBase=sys.stdout):
    """
    Will write a csv of the predicted nominal values vs the labels
    with O-C, MAE and MSE metrics, to the requested output.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :estimator: the estimator that is the source of the predictions. Used for
    field names and scales
    :row_headings: the optional heading for each row
    :selected_labels: a subset of the full list of labels/prediction names to render
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    :to: the output to write the table to. Defaults to stdout.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We plot the keys common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want names or the labels to set order
    if not selected_labels:
        selected_labels = list(estimator.label_names_and_scales.keys())
    keys = [k for k in selected_labels if k in labels[0].keys()]
    num_keys = len(keys)

    if not row_headings:
        row_headings = (f"{ix:06d}" for ix in range(len(labels)))

    # Extracts the raw values from the label and prediction List[Dict]s
    (raw_labels, pred_noms, _, ocs) \
        = _get_label_and_prediction_raw_values(labels, predictions, keys, reverse_scaling)

    # Headings row
    to.write(f"{'Target':>10s}, ")
    to.writelines(f"{k+'_lbl':>15s}, {k:>15s}, {k+'_res':>15s}, " for k in keys)
    to.write(f"{'MAE':>15s}, {'MSE':>15s}\n")

    # The instance's predictions, labels & O-Cs and its MAE & MSE
    for row_head, rlb, rnm, roc in zip(row_headings, raw_labels, pred_noms, ocs):
        row_mae, row_mse = np.mean(np.abs(roc)), np.mean(np.power(roc, 2))
        to.write(f"{row_head:>10s}, ")
        to.writelines(f"{rlb[c]:15.9f}, {rnm[c]:15.9f}, {roc[c]:15.9f}, " for c in range(num_keys))
        to.write(f"{row_mae:15.9f}, {row_mse:15.9f}\n")

    # final MAE and then MSE rows
    lbl_maes = [np.mean(np.abs(ocs[:, c])) for c in range(num_keys)]
    lbl_mses = [np.mean(np.power(ocs[:, c], 2)) for c in range(num_keys)]
    to.write(f"{'MAE':>10s}, ")
    to.writelines(f"{' '*15}, {' '*15}, {lbl_maes[c]:15.9f}, " for c in range(num_keys))
    to.write(f"{np.mean(np.abs(ocs)):15.9f}, {' '*15}\n")
    to.write(f"{'MSE':>10s}, ")
    to.writelines(f"{' '*15}, {' '*15}, {lbl_mses[c]:15.9f}, " for c in range(num_keys))
    to.write(f"{' '*15}, {np.mean(np.power(ocs, 2)):15.9f}\n")


def table_of_predictions_vs_labels(
        labels: Dict[str, float],
        predictions: Union[Dict[str, float], Dict[str, Tuple[float, float]]],
        estimator: Estimator,
        block_headings: List[str],
        selected_labels: List[str]=None,
        reverse_scaling: bool=False,
        to: TextIOBase=sys.stdout):
    """
    Will write a text table of the predicted nominal values vs the label values
    with O-C, MAE and MSE metrics, to the requested output.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :block_headings: the heading for each block of preds-vs-labels
    :estimator: the estimator that is the source of the predictions. Used for
    field names and scales
    :selected_labels: a subset of the full list of labels/prediction names to render
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    :to: the output to write the table to. Defaults to stdout.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We plot the keys common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want names or the labels to set order
    if not selected_labels:
        selected_labels = list(estimator.label_names_and_scales.keys())
    keys = [k for k in selected_labels if k in labels[0].keys()]

    # Extracts the raw values from the label and prediction List[Dict]s
    (raw_labels, pred_noms, _, ocs) \
        = _get_label_and_prediction_raw_values(labels, predictions, keys, reverse_scaling)

    line_len = 13 + (11 * len(keys))-1 + 22
    for heading, rlabs, rpreds, rocs in zip(block_headings, raw_labels, pred_noms, ocs):
        # Plot a sub table for each row of labels/predictions/ocs
        to.write("-"*line_len + "\n")
        to.write(f"{heading:<10s} | " + " ".join(f"{k:>10s}" for k in keys + ["MAE", "MSE"]))
        to.write("\n")
        rocs = np.concatenate([rocs, [np.mean(np.abs(rocs)), np.mean(np.power(rocs, 2))]])
        to.write("-"*line_len + "\n")
        for row_head, vals in zip(["Label", "Prediction", "O-C"], [rlabs, rpreds, rocs]):
            to.write(f"{row_head:<10s} | " + " ".join(f"{v:10.6f}" for v in vals))
            to.write("\n")

    # Summary rows for aggregate stats over all of the rows
    to.write("="*line_len + "\n")
    to.write(f"{'MAE':<10s} | " + " ".join(f"{v:10.6f}" for v in np.mean(np.abs(ocs), 0)) +
                 f" {np.mean(np.abs(ocs)):10.6f}\n")
    to.write(f"{'MSE':<10s} | " + " ".join([f"{v:10.6f}" for v in np.mean(np.power(ocs, 2), 0)]) +
                 " "*11 + f" {np.mean(np.power(ocs, 2)):10.6f}\n")


def plot_predictions_vs_labels(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        transit_flags: List[bool],
        plot_file: Path,
        selected_labels: List[str]=None,
        reverse_scaling: bool=False):
    """
    Will create a plot with a grid of axes, one per label, showing the
    predictions vs label values.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :transit_flags: the associated transit flags; points where the transit flag is True
    are plotted as a filled shape otherwise as an empty shape
    :plot_File: the file to save the plot to
    :selected_labels: a subset of the full list of labels/prediction names to render
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    """
    # pylint: disable=too-many-arguments, too-many-locals
    all_pub_labels = {
        "rA_plus_rB": "$r_A+r_B$",
        "k": "$k$",
        "inc": "$i$",
        "J": "$J$",
        "ecosw": r"$e\cos{\omega}$",
        "esinw": r"$e\sin{\omega}$",
        "L3": "$L_3$",
        "bP": "$b_P$"
    }

    # We plot the keys common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want names or the labels to set order
    if not selected_labels:
        selected_labels = list(all_pub_labels.keys())
    pub_labels = { k: all_pub_labels[k] for k in selected_labels if k in labels[0].keys() }

    cols = 2
    rows = math.ceil(len(pub_labels) / cols)
    _, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.9), constrained_layout=True)
    axes = axes.flatten()

    if not transit_flags:
        transit_flags = [False] * len(labels)

    print(f"Plotting scatter plot {rows}x{cols} grid for: {', '.join(pub_labels.keys())}")
    for ax_ix, (lbl_name, ax_label) in enumerate(pub_labels.items()):
        (lbl_vals, pred_vals, pred_sigmas, _) \
            = _get_label_and_prediction_raw_values(labels, predictions, [lbl_name], reverse_scaling)

        # Plot a diagonal line for exact match
        dmin, dmax = min(lbl_vals.min(), pred_vals.min()), max(lbl_vals.max(), pred_vals.max()) # pylint: disable=nested-min-max
        dmore = 0.1 * (dmax - dmin)
        diag = [dmin - dmore, dmax + dmore]
        ax = axes[ax_ix]
        ax.plot(diag, diag, color="gray", linestyle="-", linewidth=0.5)

        # Plot the preds vs labels, with those with transits highlighted
        # We want to set the fillstyle by transit flag which means plotting each item alone
        for x, y, yerr, transiting in zip(lbl_vals, pred_vals, pred_sigmas, transit_flags):
            (f, z) = ("full", 10) if transiting else ("none", 0)
            if max(np.abs(pred_sigmas)) > 0:
                ax.errorbar(x=x, y=y, yerr=yerr, fmt="o", c="tab:blue", ms=5.0, lw=1.0,
                            capsize=2.0, markeredgewidth=0.5, fillstyle=f, zorder=z)
            else:
                ax.errorbar(x=x, y=y, fmt="o", c="tab:blue", ms=5.0, lw=1.0, fillstyle=f, zorder=z)

        ax.set_ylabel(f"predicted {ax_label}")
        ax.set_xlabel(f"label {ax_label}")
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, bottom=True, left=True, right=True)
    plt.savefig(plot_file, dpi=300)
    plt.close()


def _get_label_and_prediction_raw_values(
        labels: Dict[str, float],
        predictions: Union[Dict[str, float], Dict[str, Tuple[float, float]]],
        selected_labels: List[str]=None,
        reverse_scaling: bool=False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility function which takes the List[dicts] of predictions and labels and
    extracts the raw values, calculates the O-C and returns all in a single tuple.

    :labels: the labels List[dict] to compare them with - must be in the same order as
    :predictions: the predictions List[dict] from either an Estimator or JKTEBOP
    :chosen_labels: a subset of all the label names to work with
    :reverse_scaling: whether to reapply label scaling to get the model's values
    :returns: a tuple of (label_values, prediction_values, prediction_errs, O-Cs)
    """
    if not selected_labels:
        selected_labels = list(labels[0].keys())

    # The make 2d lists for the label and prediction values
    # There are two format we expect for the predictions, either
    #   - { "key": value, "key_sigma": uncertainty }
    #   - { "key": (value, uncertainty) }
    # In either case we need to separate out the error bars
    label_values = np.array([[ldict[l] for l in selected_labels] for ldict in labels])
    if f"{selected_labels[0]}_sigma" in predictions[0]:
        nominals = np.array([[pdict[l] for l in selected_labels] for pdict in predictions])
        errors = np.array([[pdict[f"{l}_sigma"] for l in selected_labels] for pdict in predictions])
    else:
        nominals = np.array([[pdict[l][0] for l in selected_labels] for pdict in predictions])
        errors = np.array([[pdict[l][1] for l in selected_labels] for pdict in predictions])

    # Optionally reverse any scaling of the values to get them in to the scale used by the ML model
    if reverse_scaling:
        scales = [deb_example.labels_and_scales[l] for l in selected_labels]
        label_values = np.multiply(label_values, scales)
        nominals = np.multiply(nominals, scales)
        errors = np.multiply(errors, scales)

    # Currently O-C only considers the nominal values (as that's what we use in estimates)
    ocs = np.subtract(label_values, nominals)
    return (label_values, nominals, errors, ocs)


if __name__ == "__main__":
    TRAINSET_NAME = "formal-training-dataset"   # Assume the usual training set
    the_model = modelling.load_model(Path("./drop/cnn_ext_model.keras"))
    out_dir = Path(f"./drop/results/{the_model.name}/{TRAINSET_NAME}/{deb_example.pub_mags_key}")
    test_model_against_formal_test_dataset(the_model, results_dir=out_dir, plot_results=True)
