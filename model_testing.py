"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
from typing import Union, List, Dict, Tuple
from io import TextIOBase
import sys
from pathlib import Path
from contextlib import redirect_stdout
import json
import math

import matplotlib.pyplot as plt

import numpy as np
from keras.models import Model

from ebop_maven.libs import deb_example
from ebop_maven.libs.tee import Tee
from ebop_maven.estimator import Estimator
from ebop_maven import modelling

def test_with_estimator(model: Union[Model, Path],
                        test_dataset_dir: Path=Path("./datasets/formal-test-dataset"),
                        test_dataset_config: Path=Path("./config/formal-test-dataset.json"),
                        results_dir: Path=None,
                        plot_results: bool=True,
                        echo_results: bool=False):
    """
    Will test the indicated model file against the contents of the passed
    testset directory. The results for MC and non-MC estimates will be written
    to csv files in the indicated results directory.

    :model: the save model to test
    :test_dataset_dir: the location of the test dataset to use
    :test_dataset_config: Path to the config that created the test dataset
    :results_dir: the parent location to write the results csv file(s) or, if
    None, the /results/{model.name}/{trainset_name} subdirectory of the model location is used
    :plot_results: whether to produce a plot of the results vs labels
    :echo_results: whether to also echo the results csv to stdout
    """
    # Create our Estimator. It will tell us which labels it (& the model) can predict.
    estimator = Estimator(model)
    lbl_names = [*estimator.label_names_and_scales.keys()]
    num_lbls = len(estimator.label_names_and_scales)

    print(f"\nLooking for the test dataset in '{test_dataset_dir}'.")
    tfrecord_files = list(test_dataset_dir.glob("**/*.tfrecord"))
    if len(tfrecord_files) == 0:
        raise IndexError("No dataset files found")

    # Don't iterate over the dataset; it's easier if we work with all the data (via a single batch)
    # As the lbls have come through a dataset pipeline any scaling will be applied.
    map_func = deb_example.create_map_func(labels=lbl_names)
    (ds_formal_test, inst_count) = deb_example.create_dataset_pipeline(tfrecord_files,
                                                                       10000, map_func)
    (test_mags, test_feats), lbls = next(ds_formal_test.take(inst_count).as_numpy_iterator())

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

    # Run the predictions twice; with and without using MC Dropout enabled.
    for iterations, suffix in [(1, "nonmc"), (1000, "mc")]:
        # Make our prediction which will return [#inst, {}]
        # Here we don't want to unscale the predictions so we're consistent with evaluate().
        print(f"\nUsing an Estimator to make predictions on {inst_count} formal test instances.")
        predictions = estimator.predict(instance_features, iterations, unscale=False)

        # To generate the results we get the predictions (and sigmas) into shape [#inst, 8].
        noms = np.array([[d[k] for k in lbl_names] for d in predictions])

        # Summary stats (MAE, MSE) by label and over the whole set of preds.
        ocs = np.subtract(noms, lbls)
        lbl_maes = [np.mean(np.abs(ocs[:, c])) for c in range(num_lbls)]
        lbl_mses = [np.mean(np.power(ocs[:, c], 2)) for c in range(num_lbls)]
        total_mae = np.mean(np.abs(ocs))
        total_mse = np.mean(np.power(ocs, 2))

        # Now we can write out the predictions and associated losses to a csv.
        # Model name will include details of what it was trained on.
        if results_dir is None:
            parent = model.parent if isinstance(model, Path) else Path("./drop")
            results_dir = parent / f"results/{estimator.name}/unknown_training_dataset"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"formal.test.{suffix}.csv"
        output2 = sys.stdout if echo_results else None
        with redirect_stdout(Tee(open(results_file, "w", encoding="utf8"), output2)):
            # Headins row
            print(f"{'Target':>10s},",
                    *[f"{n:>15s}, {n+'_lbl':>15s}, {n+'_res':>15s}," for n in lbl_names],
                    f"{'MAE':>15s},",
                    f"{'MSE':>15s}")

            # The instance's predictions, labels & O-Cs and its MAE & MSE
            for target, rnom, rlbl, roc in zip(target_names, noms, lbls, ocs):
                row_mae, row_mse = np.mean(np.abs(roc)), np.mean(np.power(roc, 2))
                print(f"{target:>10s},",
                   *[f"{rnom[c]:15.9f}, {rlbl[c]:15.9f}, {roc[c]:15.9f}," for c in range(num_lbls)],
                    f"{row_mae:15.9f},",
                    f"{row_mse:15.9f}")

            # Loss rows, one each for MAE & MSE with grand totals at the end
            print(f"{'MAE':>10s},",
                *[f"{' '*15}, {' '*15}, {lbl_maes[c]:15.9f}," for c in range(num_lbls)],
                f"{total_mae:15.9f},",
                f"{' '*15}")
            print(f"{'MSE':>10s},",
                *[f"{' '*15}, {' '*15}, {lbl_mses[c]:15.9f}," for c in range(num_lbls)],
                f"{' '*15},",
                f"{total_mse:15.9f}")

        print(f"\nSaved predictions and associated loss information to '{results_file}'")
        print("--------------------------------")
        print(f"Total MAE ({suffix}): {total_mae:.9f}")
        print(f"Total MSE ({suffix}): {total_mse:.9f}")
        print("--------------------------------\n")

        labels = [dict(zip(lbl_names, lbl_row)) for lbl_row in lbls]
        with open(results_file.parent / f"{results_file.stem}.txt", "w", encoding="utf8") as of:
            table_of_labels_vs_predictions(labels, predictions, target_names, estimator, to=of)
        if plot_results:
            plot_file = results_file.parent / f"predictions_vs_labels_{suffix}.eps"
            plot_predictions_vs_labels(labels, predictions, transit_flags, plot_file)


def plot_predictions_vs_labels(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        transit_flags: List[bool],
        plot_file: Path,
        names: List[str]=None,
        rescale: bool=False):
    """
    Will create a plot with a grid of axes, one per label, showing the
    predictions vs label values.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :transit_flags: the associated transit flags; points where the transit flag is True
    are plotted as a filled shape otherwise as an empty shape
    :plot_File: the file to save the plot to
    :names: a subset of the full list of labels/prediction names to render
    :rescale: whether the re-scale the values so they represent the underlying model output
    """
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

    # Use list and order of label names or the dict above if not given
    chosen_labels = names if names else all_pub_labels.keys()
    pub_labels = { k: all_pub_labels[k] for k in chosen_labels if k in labels[0].keys() }

    cols = 2
    rows = math.ceil(len(pub_labels) / cols)
    _, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.9), constrained_layout=True)
    axes = axes.flatten()

    # Produce a plot of prediction vs label for each label type
    print(f"Plotting scatter plot {rows}x{cols} grid for: {', '.join(pub_labels.keys())}")
    for ax_ix, (lbl_name, ax_label) in enumerate(pub_labels.items()):
        lbl_vals = [lbls[lbl_name] for lbls in labels]
        # There are two format we expect for the predictions, either
        #   - { "key": value, "key_sigma": sigma_value }    (from Estimator)
        #   - { "key": (value, sigma_value) }               (from JKTEBOP prediction)
        # In either case we need to separate out the sigmas
        if f"{lbl_name}_sigma" in predictions[0]:
            pred_vals = [preds[lbl_name] for preds in predictions]
            pred_sigmas = [preds[f"{lbl_name}_sigma"] for preds in predictions]
        else:
            pred_vals = [preds[lbl_name][0] for preds in predictions]
            pred_sigmas = [preds[lbl_name][1] for preds in predictions]

        if rescale:
            scale = deb_example.labels_and_scales[lbl_name]
            lbl_vals = np.multiply(lbl_vals, scale)
            pred_vals = np.multiply(pred_vals, scale)
            pred_sigmas = np.multiply(pred_sigmas, scale)

        # Plot a diagonal line for exact match
        dmin, dmax = min(min(lbl_vals), min(pred_vals)), max(max(lbl_vals), max(pred_vals)) # pylint: disable=nested-min-max
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


def table_of_labels_vs_predictions(
        labels: Dict[str, float],
        predictions: Union[Dict[str, float], Dict[str, Tuple[float, float]]],
        titles: List[str],
        estimator: Estimator,
        names: List[str]=None,
        rescale: bool=False,
        to: TextIOBase=sys.stdout):
    """
    Will write a text table of the labels vs predictes nominal values
    with O-C, MAE and MSE metrics, to the requested output.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :titles: the title for each row
    :estimator: the estimator that is the source of the predictions. Used for
    field names and scales
    :names: a subset of the full list of labels/prediction names to render
    :rescale: whether the re-scale the values so they represent the underlying model output
    :to: the output to write the table to. Defaults to stdout.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We plot the keys common to the labels & preds, & optionally the input list
    # of names. Avoiding using set() as we want names or the labels to set order
    keys = names if names else estimator.label_names_and_scales.keys()
    keys = [k for k in keys if k in labels[0].keys()]

    # The make 2d lists for the label and pred values so we can perform matrix calcs
    # There are two format we expect for the predictions, either
    #   - { "key": value, "key_sigma": sigma_value }
    #   - { "key": (value, sigma_value) }
    # In either case we need to separate out the sigmas
    raw_labels = [[label_dict[k] for k in keys] for label_dict in labels]
    if predictions and isinstance(predictions[0][keys[0]], tuple):
        pred_noms = [[pred_dict[k][0] for k in keys] for pred_dict in predictions]
        pred_sigmas = [[pred_dict[k][1] for k in keys] for pred_dict in predictions]
    else:
        pred_noms = [[pred_dict[k] for k in keys] for pred_dict in predictions]
        pred_sigmas = [[pred_dict[f"{k}_sigma"] for k in keys] for pred_dict in predictions]

    # Optionally reapply the scaling the model uses so metrics represent the model not the Estimator
    if rescale:
        scales = [estimator.label_names_and_scales[k] for k in keys]
        raw_labels = np.multiply(raw_labels, scales)
        pred_noms = np.multiply(pred_noms, scales)
        pred_sigmas = np.multiply(pred_sigmas, scales)

    # Currently O-C only considers the nominal values (as that's what we use)
    ocs = np.subtract(raw_labels, pred_noms)

    line_len = 13 + (11 * len(keys))-1 + 22

    for title, rlabs, rpreds, rocs in zip(titles, raw_labels, pred_noms, ocs):
        # Plot a sub table for each row of labels/predictions/ocs
        to.write("-"*line_len + "\n")
        to.write(f"{title:<10s} | " + " ".join(f"{k:>10s}" for k in keys + ["MAE", "MSE"]))
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


if __name__ == "__main__":
    TRAINSET_NAME = "formal-training-dataset"   # Assume the usual training set
    the_model = modelling.load_model(Path("./drop/cnn_ext_model.keras"))
    out_dir = Path(f"./drop/results/{the_model.name}/{TRAINSET_NAME}/{deb_example.pub_mags_key}")
    test_with_estimator(the_model, results_dir=out_dir, plot_results=True)
