"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
from typing import Union, List
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
    num_labels = len(estimator.label_names_and_scales)

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

    # Run the predictions twice; with and without using MC Dropout enabled.
    for iterations, suffix in [(1, "nonmc"), (1000, "mc")]:
        # Make our prediction which will return [#inst, {}]
        # Here we don't want to unscale the predictions so we're consistent with evaluate().
        print(f"\nUsing an Estimator to make predictions on {inst_count} formal test instances.")
        predictions = estimator.predict(instance_features, iterations, unscale=False)

        # To generate the results we get the predictions (and sigmas) into shape [#inst, 8].
        noms = np.array([[d[k] for k in lbl_names] for d in predictions])
        nom_errs = np.array([[d[f"{k}_sigma"] for k in lbl_names] for d in predictions])

        # Summary stats (MAE, MSE) by label and over the whole set of preds. 
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

            # Read additional target data from the config file
            transit_flags = [False] * inst_count
            with open(test_dataset_config, mode="r", encoding="utf8") as f:
                targets = json.load(f)
                transit_flags = [config.get("transits", False) for t, config in targets.items()]

            pub_fields = { k: v for (k, v) in all_pub_labels.items() if k in lbl_names }
            rows = math.ceil(len(pub_fields) / cols)
            _, axes = plt.subplots(rows, cols,
                                   figsize=(cols*size, rows*size), constrained_layout=True)
            axes = axes.flatten()
            print(f"Plotting single publication scatter plot {rows}x{cols} grid",
                  f"for the fields: {', '.join(pub_fields.keys())}")

            # Produce a plot of prediction vs label for each label type
            for ax_ix, (lbl_name, ax_label) in enumerate(pub_fields.items()):
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
                # We want to set the fillstyle by transit flag which means plotting each item alone
                for x, y, yerr, transiting in zip(lbl_vals, pred_vals, pred_errs, transit_flags):
                    (fill, z) = ("full", 10) if transiting else ("none", 0)
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
    TRAINSET_NAME = "formal-training-dataset"   # Assume the usual training set
    the_model = modelling.load_model(Path("./drop/cnn_ext_model.keras"))
    out_dir = Path(f"./drop/results/{the_model.name}/{TRAINSET_NAME}/{deb_example.pub_mags_key}")
    test_with_estimator(the_model, results_dir=out_dir, plot_results=True)
