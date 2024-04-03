"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
from typing import Union, List
import sys
from pathlib import Path
from contextlib import redirect_stdout
import math

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from keras.models import Model

from ebop_maven.libs import deb_example
from ebop_maven.libs.tee import Tee
from ebop_maven.estimator import Estimator

def test_with_estimator(model: Union[Model, Path],
                        test_dataset_dir: Path,
                        results_dir: Path=None,
                        plot_results: bool=True,
                        echo_results: bool=False,
                        excluded_labels: List[str]=None):
    """
    Will test the indicated model file against the contents of the passed
    testset directory. The results for MC and non-MC estimates will be written
    to csv files in the indicated results directory.

    :model: the save model to test
    :test_dataset_dir: the location of the test dataset to use
    :results_dir: the parent location to write the results csv file(s) or, if
    None, the /results/{model.name}/{trainset_name} subdirectory of the model location is used
    :plot_results: whether to produce a plot of the results vs labels
    :echo_results: whether to also echo the results csv to stdout
    :excluded_labels: list of labels to exclude from the report.
    """
    print(f"\nLooking for the test dataset in '{test_dataset_dir}'.")
    tfrecord_files = list(test_dataset_dir.glob("**/*.tfrecord"))
    if len(tfrecord_files) == 0:
        raise IndexError("No dataset files found")

    ds_formal_test = tf.data.TFRecordDataset(tfrecord_files)
    test_count = ds_formal_test.reduce(0, lambda count, _: count+1).numpy()

    print(f"Found {len(tfrecord_files)} tfrecord file(s) containing {test_count} instances.")
    ds_formal_test = ds_formal_test.map(deb_example.map_parse_deb_example).batch(test_count)

    # Don't iterate over the dataset; it's easier if we work with all the data
    (test_lcs, test_feats), lbls = next(ds_formal_test.take(test_count).as_numpy_iterator())

    # The features arrive in a tuple (array[shape(#inst, #bin, 1)], array[shape(#inst, #feat, 1)])
    # We need them in the input format for the Estimatory [{"lc":.. , "feat1": ... }]
    fn = [*deb_example.extra_features_and_defaults.keys()]
    instance_features = [
        { "lc": tl, **{n: f[0] for (n, f) in zip(fn, tf)} } for tl, tf in zip(test_lcs, test_feats)
    ]

    # Get the details of the labels we're to report on.
    # Optionally, discard any we're no longer interested in.
    lbl_names, lbl_scales = deb_example.label_names.copy(), deb_example.label_scales.copy()
    if excluded_labels is not None:
        for exclude_label in excluded_labels:
            exclude_ix = lbl_names.index(exclude_label)
            lbl_names.pop(exclude_ix)
            lbl_scales.pop(exclude_ix)
            lbls = np.delete(lbls, exclude_ix, 1)
    num_labels = len(lbl_names)

    # Run the predictions twice; with and without using MC Dropout enabled.
    for iterations, suffix in [(1, "nonmc"), (1000, "mc")]:
        # Make our prediction which will return [#inst, {}]
        print(f"\nUsing an Estimator to make predictions on {test_count} formal test instances.")
        estimator = Estimator(model, iterations)
        # training_set = estimator.attrs.get("training_dataset", "unkown")
        # print(f"The model was trained on the '{training_set}' training set")
        # print()
        predictions = estimator.predict(instance_features)

        # To generate the results we get the predictions (not sigmas) into
        # shape [#inst, 8]. The labels/predictions are scaled in the Model which
        # the estimator's funcs handle (i.e. it's trained on inc/100). However
        # here we un-scale the predictions so we're consistent with evaluate().
        noms = np.array([[d[k] for k in lbl_names] for d in predictions])
        noms = np.multiply(noms, lbl_scales)
        nom_errs = np.array([[d[f"{k}_sigma"] for k in lbl_names] for d in predictions])
        nom_errs = np.multiply(nom_errs, lbl_scales)

        # Summary stats (MAE, MSE) by label and over the whole set of preds
        ocs = np.subtract(noms, lbls)
        lbl_maes = [np.mean(np.abs(ocs[:, c])) for c in range(num_labels)]
        lbl_mses = [np.mean(np.power(ocs[:, c], 2)) for c in range(num_labels)]
        total_mae = np.mean(np.abs(ocs))
        total_mse = np.mean(np.power(ocs, 2))

        # Now we can write out the predictions and associated losses to a csv.
        # Model name will include details of what it was trained on.
        if results_dir is None:
            parent = model.parent if isinstance(model, Path) else Path("./drop")
            td_name = estimator.attrs.get("training_dataset", "unknown_training_dataset")
            results_dir = parent / f"results/{estimator.name}/{td_name}"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"formal.test.{suffix}.csv"
        output2 = sys.stdout if echo_results else None
        with redirect_stdout(Tee(open(results_file, "w", encoding="utf8"), output2)):
            # Headins row
            print("Inst,",
                    *[f"{n:>15s}, {n+'_lbl':>15s}, {n+'_res':>15s}," for n in lbl_names],
                    f"{'MAE':>15s},",
                    f"{'MSE':>15s}")

            # The instance's predictions, labels & O-Cs and its MAE & MSE
            for r in range(ocs.shape[0]):
                row_mae, row_mse = np.mean(np.abs(ocs[r])), np.mean(np.power(ocs[r], 2))
                print(f"{r+1:4d},",
                    *[f"{noms[r, c]:15.9f}, {lbls[r, c]:15.9f}, {ocs[r, c]:15.9f},"
                                                                        for c in range(num_labels)],
                    f"{row_mae:15.9f},",
                    f"{row_mse:15.9f}")

            # Loss rows, one each for MAE & MSE with grand totals at the end
            print(" MAE,",
                *[f"{' '*15}, {' '*15}, {lbl_maes[c]:15.9f}," for c in range(num_labels)],
                f"{total_mae:15.9f},",
                f"{' '*15}")
            print(" MSE,",
                *[f"{' '*15}, {' '*15}, {lbl_mses[c]:15.9f}," for c in range(num_labels)],
                f"{' '*15},",
                f"{total_mse:15.9f}")

        print(f"\nSaved predictions and associated loss information to '{results_file}'")
        print("--------------------------------")
        print(f"Total MAE ({suffix}): {total_mae:.9f}")
        print(f"Total MSE ({suffix}): {total_mse:.9f}")
        print("--------------------------------\n")

        if plot_results:
            size = 3
            cols = 2
            pub_fields = ["rA_plus_rB", "k", "inc", "J",
                          "ecosw", "esinw", "L3", "bP"]
            pub_labels = ["$r_A+r_B$", "$k$", "$i$", "$J$",
                          r"$e\cos{\omega}$", r"$e\sin{\omega}$", "$L_3$", "$b_P$"]
            rows = math.ceil(len(pub_fields) / cols)
            _, axes = plt.subplots(rows, cols,
                                   figsize=(cols*size, rows*size), constrained_layout=True)
            axes = axes.flatten()
            print(f"Plotting single publication scatter plot {rows}x{cols} grid",
                  f"for the fields: {pub_fields}")

            # Produce a plot of prediction vs label for each label type
            for ax_ix, (lbl_name, ax_label) in enumerate(zip(pub_fields, pub_labels)):
                c = lbl_names.index(lbl_name)
                lbl_vals = lbls[:, c]
                pred_vals = noms[:, c]
                pred_errs = nom_errs[:, c]

                # Plot a diagonal line for exact match
                ax = axes[ax_ix]
                diag = [min(lbl_vals.min(), pred_vals.min()) - 0.1,
                        max(lbl_vals.max(), pred_vals.max()) + 0.1]
                ax.plot(diag, diag, color="gray", linestyle="-", linewidth=0.5)

                # Plot the preds vs labels, with those with transits highlighted
                # TODO: this is a bit of a hack for now. We need some way of
                # passing transiting/eclipsing flag through the dataset
                transit_ixs = [7, 16, 17]
                for fill, z, plot_ixs in [
                    ("none", 0, [i for i in range(lbl_vals.shape[0]) if i not in transit_ixs]),
                    ("full", 10, transit_ixs),
                ]:
                    x = [lbl_vals[i] for i in plot_ixs]
                    y = [pred_vals[i] for i in plot_ixs]
                    yerr = [pred_errs[i] for i in plot_ixs]

                    if suffix == "mc":
                        ax.errorbar(x=x, y=y, yerr=yerr, fmt="o", c="tab:blue", ms=5.0, lw=1.0,
                                    capsize=2.0, markeredgewidth=0.5, fillstyle=fill, zorder=z)
                    else:
                        ax.errorbar(x=x, y=y, fmt="o", c="tab:blue", ms=5.0, lw=1.0,
                                    fillstyle=fill, zorder=z)

                ax.set_ylabel(f"predicted {ax_label}")
                ax.set_xlabel(f"label {ax_label}")
                ax.tick_params(axis="both", which="both", direction="in",
                               top=True, bottom=True, left=True, right=True)
            plt.savefig(results_dir / f"predictions_vs_labels_{suffix}.eps")
            plt.close()

if __name__ == "__main__":
    MODEL_FILE_NAME = "cnn_ext_model"
    test_with_estimator(model=Path(f"./drop/{MODEL_FILE_NAME}.keras"),
                        test_dataset_dir= Path("./datasets/formal-test-dataset/1024/wm-0.75/"),
                        results_dir=None,
                        plot_results=True,
                        echo_results=False,
                        excluded_labels=[])
