"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
# pylint: disable=too-many-arguments, too-many-locals, no-member, import-error
from typing import Union, List, Dict, Tuple
from io import TextIOBase, StringIO
import sys
from pathlib import Path
import json
import math
import re
from contextlib import redirect_stdout
from textwrap import TextWrapper

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model

from ebop_maven.libs.tee import Tee
from ebop_maven.libs import deb_example, lightcurve, jktebop, stellar, limb_darkening
from ebop_maven.estimator import Estimator
from ebop_maven import modelling, datasets

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
    to the config for the formal test dataset so that it can retrieve flags.

    :model: the save model to test
    :test_dataset_dir: the location of the formal test dataset
    :test_dataset_config: Path to the config that created the test dataset
    :results_dir: the parent location to write the results csv file(s) or, if
    None, the /results/{model.name}/{trainset_name} subdirectory of the model location is used
    :plot_results: whether to produce a plot of the results vs labels
    """
    # Create our Estimator. It will tell us which labels it (& the model) can predict.
    estimator = Estimator(model)
    l_names = list(estimator.label_names_and_scales.keys())
    f_names = estimator.input_feature_names

    ids, labels, features = \
        _get_dataset_labels_and_features(test_dataset_dir, estimator, l_names, f_names, True)
    inst_count = len(ids)

    # Read additional target data from the config file
    t_names = [i.split("/")[0] for i in ids] # formal test set has these as target/sector
    with open(test_dataset_config, mode="r", encoding="utf8") as f:
        targets_config = json.load(f)
        transit_flags = [targets_config.get(tn, {}).get("transits", False) for tn in t_names]

    # Make the results directory
    if results_dir is None:
        parent = model.parent if isinstance(model, Path) else Path("./drop")
        results_dir = parent / f"results/{estimator.name}/formal-training-dataset"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run the predictions twice; with and without using MC Dropout enabled.
    for iterations, suffix in [(1, "nonmc"), (1000, "mc")]:
        # Make our prediction which will return [{"name": value, "name_sigma": sigma_value}]*insts
        # We don't want to unscale predictions so we get the values predicted by the model,
        # and which match the labels & are consistent with model.evaluate().
        print(f"\nUsing an Estimator to make predictions on {inst_count} formal test instances.")
        predictions = estimator.predict(features, iterations, unscale=False)

        results_file = results_dir / f"formal.test.{suffix}.csv"
        with open(results_file, mode="w", encoding="utf8") as of:
            predictions_vs_labels_to_csv(labels, predictions, estimator, ids, l_names, to=of)

        print(f"\nSaved predictions and associated loss information to '{results_file}'")
        (_, _, _, ocs) = _get_label_and_prediction_raw_values(labels, predictions)
        print("--------------------------------")
        print(f"Total MAE ({suffix}): {np.mean(np.abs(ocs)):.9f}")
        print(f"Total MSE ({suffix}): {np.mean(np.power(ocs, 2)):.9f}")
        print("--------------------------------\n")

        with open(results_file.parent / f"{results_file.stem}.txt", "w", encoding="utf8") as of:
            table_of_predictions_vs_labels(labels, predictions, estimator, ids, l_names, to=of)

        if plot_results:
            plot_file = results_file.parent / f"predictions_vs_labels_{suffix}.eps"
            plot_predictions_vs_labels(labels, predictions, transit_flags, plot_file)


def test_fitting_against_formal_test_dataset(
        model: Union[Model, Path],
        results_dir: Path,
        mc_iterations: int=1,
        skip_ids: List[str]=None,
        apply_fit_overrides: bool=False,
        fit_with_labels: bool=False,
        plot_results: bool=True,
        test_dataset_dir: Path=Path("./datasets/formal-test-dataset"),
        test_dataset_config: Path=Path("./config/formal-test-dataset.json")):
    """
    Will test the indicated model file by making predictions against the formal
    test dataset and then using these predictions to fit the corresponding
    lightcurves with JKTEBOP task 3. The results of the fitting will be written
    to the result directory. Will also need access to the config for the formal
    test dataset so that it can correctly set up the lightcurves for fitting.

    :model: the save model to test
    :results_dir: the parent location to write the results csv file(s) or, if
    None, the /results/{model.name}/{trainset_name} subdirectory of the model location is used
    :mc_iterations: the number of MC Dropout iterations
    :skip_ids: list of target ids to not fit
    :apply_fit_overrides: apply any fit_overrides from the target's config
    :fit_with_labels: if True we'll use the label as inputs to the fit, not estimates,
    so that we can test whether a good fit is possible provided the inputs are good enough
    :plot_results: whether to produce a plot of the results vs labels
    :test_dataset_dir: the location of the formal test dataset
    :test_dataset_config: Path to the config that created the test dataset
    """
    # Create our Estimator. It will tell us which labels it (& the model) can predict.
    estimator = Estimator(model)
    l_names = list(estimator.label_names_and_scales.keys())
    f_names = estimator.input_feature_names
    results_dir.mkdir(parents=True, exist_ok=True)
    tw = TextWrapper(100)

    # Gets the target details, labels (unscaled) and the mags/ext features make preds from
    fitted_params = []
    targets, labels, features = _get_dataset_labels_and_features(
                                    test_dataset_dir, estimator, l_names, f_names, False, skip_ids)
    target_count = len(targets)
    with open(test_dataset_config, mode="r", encoding="utf8") as f:
        targets_config = json.load(f)

    # Estimator prediction will return [{"name": value, "name_sigma": sigma_value}]*insts
    # We'll be using these is inputs to fitting so we need to let estimator do its scaling.
    for ix, (target, t_labels, t_feats) in enumerate(zip(targets, labels, features)):
        target_cfg = targets_config[target]
        print(f"\n\nProcessing target {ix+1} of {target_count}: {target}\n" + "-"*40)
        print(tw.fill(target_cfg.get("desc", "")) + "\n")

        # The basic lightcurve data read, rectified & extended with delta_mag and delta_mag_err cols
        (lc, sector_count) = datasets.prepare_lightcurve_for_target(target, target_cfg, True)
        pe = lightcurve.to_lc_time(target_cfg["primary_epoch"], lc).value
        period = target_cfg["period"]

        file_stem = f"model-testing-{re.sub(r'[^\w\d-]', '-', target.lower())}"
        file_parent = jktebop.get_jktebop_dir()
        for file in file_parent.glob(f"{file_stem}.*"):
            file.unlink()
        in_fname = file_parent / f"{file_stem}.in"
        dat_fname = file_parent / f"{file_stem}.dat"

        if not fit_with_labels:
            print("\nUsing the Estimator's unscaled predictions as the input params to fit", target)
            input_params = estimator.predict([t_feats], iterations=mc_iterations)[0]
        else:
            print(f"\n***Warning: Using the labels as the input params to fit {target}***")
            input_params = t_labels
        table_of_predictions_vs_labels([t_labels], [input_params], estimator, [target])

        # published fitting params that may be needed for good fit
        fit_overrides = target_cfg.get("fit_overrides", {}) if apply_fit_overrides else {}
        lrats = fit_overrides.pop("lrat", [])

        params = {
            **base_jktebop_task3_params(period, pe, dat_fname.name, file_stem, target_cfg),
            **input_params,
            **fit_overrides,
        }

        # Add scale-factor poly fitting, chi^2 adjustment (to 1.0) or light-ratio instructions
        segments = lightcurve.find_lightcurve_segments(lc, 0.5, return_times=True)
        append_lines = jktebop.build_poly_instructions(segments, "sf", 1)
        append_lines += ["", "chif", ""] + [ f"lrat {l}" for l in lrats ]

        jktebop.write_in_file(in_fname, task=3, append_lines=append_lines, **params)
        jktebop.write_light_curve_to_dat_file(
                    lc, dat_fname, column_formats=[lambda t: f"{t.value:.6f}", "%.6f", "%.6f"])

        # Don't consume the output files so they're available for subsequent plotting
        print(f"\nFitting {target} (with {sector_count} sector(s) of data) using JKTEBOP task 3...")
        par_fname = file_parent / f"{file_stem}.par"
        par_contents = list(jktebop.run_jktebop_task(in_fname, par_fname, stdout_to=sys.stdout))
        t_params = jktebop.read_fitted_params_from_par_lines(par_contents, l_names)
        table_of_predictions_vs_labels([t_labels], [t_params], estimator, [target])
        fitted_params.append(t_params)

    output_stem = "fitted_params_vs_labels" + "_control" if fit_with_labels else ""
    with open(results_dir / f"{output_stem}.txt", "w", encoding="utf8") as of:
        table_of_predictions_vs_labels(labels, fitted_params, estimator, targets, l_names, to=of)

    if plot_results:
        transit_flags = [targets_config[tn].get("transits", False) for tn in targets]
        plot_file = results_dir / f"{output_stem}.eps"
        plot_predictions_vs_labels(labels, fitted_params, transit_flags, plot_file)


def base_jktebop_task3_params(period: float,
                              primary_epoch: float,
                              dat_file_name: str,
                              file_name_stem: str,
                              sector_cfg: Dict[str, any]) -> Dict[str, any]:
    """
    Get the basic testing set of JKTEBOP task3 in file parameters.
    This sets up mainly fixed values for qphot, grav darkening etc.

    However, quad limb darkening algo and coeffs are found by lookup based
    on the stars' masses, radii and effective temps
    
    L3 is taken from the labels as we currently don't estimate this.
    If the L3 is zero we set it to fixed too.
    """
    # Calculate star specific LD params
    ld_params = {}
    for star in ["A", "B"]:
        logg = stellar.log_g(sector_cfg[f"M{star}"] * u.solMass, sector_cfg[f"R{star}"] * u.solRad)
        coeffs = limb_darkening.lookup_tess_quad_ld_coeffs(logg, sector_cfg[f"Teff{star}"] * u.K)
        ld_params[f"LD{star}"] = "quad"
        ld_params[f"LD{star}1"] = coeffs[0]
        ld_params[f"LD{star}2"] = coeffs[1]

    l3 = sector_cfg["labels"].get("L3", 0)
    l3_fit = 0 if l3 == 0 else 1
    if not jktebop.get_jktebop_support_neg_l3():
        l3 = max(0, l3)

    return {
        "qphot": 0.,
        "gravA": 0.,        "gravB": 0.,
        "L3": l3,

        **ld_params,

        "reflA": 0.,        "reflB": 0.,
        "period": period,
        "primary_epoch": primary_epoch,

        "ecosw_fit": 1,     "esinw_fit": 1,
        "L3_fit": l3_fit,
        "LDA1_fit": 1,      "LDB1_fit": 1,
        "LDA2_fit": 0,      "LDB2_fit": 0,
        "period_fit": 1,
        "primary_epoch_fit": 1,
        "data_file_name": dat_file_name,
        "file_name_stem": file_name_stem,
    }


def predictions_vs_labels_to_csv(
        labels: Dict[str, float],
        predictions: Union[Dict[str, float], Dict[str, Tuple[float, float]]],
        estimator: Estimator,
        row_headings: List[str]=None,
        selected_labels: List[str]=None,
        reverse_scaling: bool=False,
        to: TextIOBase=None):
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

    print_it = not to
    if print_it:
        to = StringIO()

    # Headings row
    to.write(f"{'Target':>12s}, ")
    to.writelines(f"{k+'_lbl':>15s}, {k:>15s}, {k+'_res':>15s}, " for k in keys)
    to.write(f"{'MAE':>15s}, {'MSE':>15s}\n")

    # The instance's predictions, labels & O-Cs and its MAE & MSE
    for row_head, rlb, rnm, roc in zip(row_headings, raw_labels, pred_noms, ocs):
        row_mae, row_mse = np.mean(np.abs(roc)), np.mean(np.power(roc, 2))
        to.write(f"{row_head:>12s}, ")
        to.writelines(f"{rlb[c]:15.9f}, {rnm[c]:15.9f}, {roc[c]:15.9f}, " for c in range(num_keys))
        to.write(f"{row_mae:15.9f}, {row_mse:15.9f}\n")

    # final MAE and then MSE rows
    lbl_maes = [np.mean(np.abs(ocs[:, c])) for c in range(num_keys)]
    lbl_mses = [np.mean(np.power(ocs[:, c], 2)) for c in range(num_keys)]
    to.write(f"{'MAE':>12s}, ")
    to.writelines(f"{' '*15}, {' '*15}, {lbl_maes[c]:15.9f}, " for c in range(num_keys))
    to.write(f"{np.mean(np.abs(ocs)):15.9f}, {' '*15}\n")
    to.write(f"{'MSE':>12s}, ")
    to.writelines(f"{' '*15}, {' '*15}, {lbl_mses[c]:15.9f}, " for c in range(num_keys))
    to.write(f"{' '*15}, {np.mean(np.power(ocs, 2)):15.9f}\n")
    if print_it:
        print(to.getvalue())


def table_of_predictions_vs_labels(
        labels: Dict[str, float],
        predictions: Union[Dict[str, float], Dict[str, Tuple[float, float]]],
        estimator: Estimator,
        block_headings: List[str],
        selected_labels: List[str]=None,
        reverse_scaling: bool=False,
        to: TextIOBase=None):
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
    :to: the output to write the table to. Defaults to printing.
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

    print_it = not to
    if print_it:
        to = StringIO()

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
    if print_it:
        print(to.getvalue())


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


def _get_dataset_labels_and_features(
    dataset_dir: Path,
    estimator: Estimator,
    label_names: List[str]=None,
    feature_names: List[str]=None,
    scale_labels: bool=True,
    skip_ids: List[str]=None):
    """
    Gets the ids, labels and features of the requested dataset.

    :dataset_dir: the directory within which it lives
    :estimator: Estimator instance which publishes the full set of label/feature names supported
    :label_names: the names of the labels to return, or all suuported by estimatory if None
    :feature_names: the names of the features to return, or all suuported by estimatory if None
    :scale_labels: if True labels will be scaled
    :skip_ids: List of ids to match on which we skip
    :returns: Tuple[List[ids], List[labels dict], List[features dict]]
    """
    if not label_names:
        label_names = list(estimator.label_names_and_scales.keys())
    if not feature_names:
        feature_names = estimator.input_feature_names

    print(f"Looking for the test dataset in '{dataset_dir}'...", end="")
    tfrecord_files = list(dataset_dir.glob("**/*.tfrecord"))
    if len(tfrecord_files) > 0:
        print(f"found {len(tfrecord_files)} dataset file(s).")
    else:
        raise IndexError("No dataset files found")

    ids, labels, features = [], [], []
    for (targ, lrow, mrow, frow) in datasets.inspect_dataset(tfrecord_files,
                                                             scale_labels=scale_labels):
        if not skip_ids or targ not in skip_ids:
            ids += [targ]
            labels += [{ ln: lrow[ln] for ln in label_names }]
            features += [{
                "mags": mrow[deb_example.pub_mags_key], 
                **{ fn: frow[fn] for fn in feature_names if fn not in ["mags"] }}
            ]
    return ids, labels, features


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
    #   - { "key": (value, uncertainty) }
    #   - { "key": value, "key_sigma": uncertainty }
    # In either case we need to separate out the error bars
    label_values = np.array([[ld[l] for l in selected_labels] for ld in labels])
    if isinstance(predictions[0][selected_labels[0]], tuple):
        nominals = np.array([[pd[l][0] for l in selected_labels] for pd in predictions])
        errors = np.array([[pd[l][1] for l in selected_labels] for pd in predictions])
    else:
        nominals = np.array([[pd[l] for l in selected_labels] for pd in predictions])
        if f"{selected_labels[0]}_sigma" in predictions[0]:
            errors = np.array([[pd[f"{l}_sigma"] for l in selected_labels] for pd in predictions])
        else:
            errors = np.zeros_like(nominals)

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
    out_dir.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(Tee(open(out_dir / "model_testing.log", "w", encoding="utf8"))):
        the_model = modelling.load_model(Path("./drop/cnn_ext_model.keras"))

        print("\nTesting the model's estimates\n" + "="*29)
        test_model_against_formal_test_dataset(the_model, results_dir=out_dir, plot_results=True)

        # Now look at fitting the format-test-dataset
        skip = ["V402 Lac", "V456 Cyg"] # Neither are suitable for JKTEBOP fitting

        # Uses the estimator/model inputs to the fitting
        print("\n\nTesting the JKTEBOP fitting derived from the model's estimates\n" + "="*62)
        test_fitting_against_formal_test_dataset(the_model, out_dir, 1,
                                                 skip_ids=skip,
                                                 apply_fit_overrides=True,
                                                 fit_with_labels=False)

        # Use the labels as the inputs so we have control data
        print("\n\nTesting the JKTEBOP fitting using the labels as inputs\n" + "="*54)
        test_fitting_against_formal_test_dataset(the_model, out_dir, 1,
                                                 skip_ids=skip,
                                                 apply_fit_overrides=True,
                                                 fit_with_labels=True)
