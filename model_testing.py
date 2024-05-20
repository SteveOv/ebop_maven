"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
# pylint: disable=too-many-arguments, too-many-locals, no-member, import-error
from typing import Union, List, Dict, Tuple
from io import TextIOBase, StringIO
import sys
from pathlib import Path
import json
import re
from contextlib import redirect_stdout
from textwrap import TextWrapper, fill
import copy
import argparse

from uncertainties import ufloat
from uncertainties.umath import sqrt, acos, asin, degrees # pylint: disable=no-name-in-module

import astropy.units as u
import numpy as np
from keras import Model, metrics

from ebop_maven.libs.tee import Tee
from ebop_maven.libs import deb_example, lightcurve, jktebop, stellar, limb_darkening
from ebop_maven.estimator import Estimator
from ebop_maven import datasets
import plots

def test_model_against_formal_test_dataset(
        estimator: Union[Model, Estimator],
        mc_iterations: int=1,
        include_ids: List[str]=None,
        scaled: bool=False,
        test_dataset_dir: Path=Path("./datasets/formal-test-dataset")) \
            -> Tuple[np.ndarray[Dict[str, float]], np.ndarray[Dict[str, float]]]:
    """
    Will test the indicated model file or Estimator against the contents of the
    formal test dataset.

    :use_estimator: the Estimator or estimator model to use to make predictions
    :mc_iterations: the number of MC Dropout iterations
    :include_ids: list of target ids to predict, or all if not set
    :scaled: whether labels and predictions are scaled (like raw model predictions) or not
    :test_dataset_dir: the location of the formal test dataset
    :returns: a tuple of (List[labels dicts], List[predictions dicts])
    one row per target instance in the order of the dataset
    """
    # Create our Estimator. It will tell us what its inputs should look like
    # and which labels it (& the underlying model) can predict.
    if not isinstance(estimator, Estimator):
        estimator = Estimator(estimator)

    # The estimator may not predict inc but we will need it later when fitting
    label_names = estimator.label_names
    calculated_inc = "inc" not in label_names
    if calculated_inc:
        label_names += ["inc"]

    # Gets the target ids (names), labels and mags/ext features to predict on
    ids, labels, features = get_dataset_labels_and_features(
                test_dataset_dir,
                label_names=label_names,
                feature_names=estimator.input_feature_names,
                mags_bins=estimator.mags_feature_bins,
                mags_wrap_phase=estimator.mags_feature_wrap_phase,
                scaled_labels=scaled,
                include_ids=include_ids)
    if include_ids is not None:
        assert len(include_ids) == len(ids)

    # Make our prediction which will return [{"name": value, "name_sigma": sigma_value}]*insts
    print(f"\nThe Estimator is making predictions on the {len(ids)} formal test instances",
          f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
    predictions = estimator.predict(features, mc_iterations, unscale=not scaled)
    if calculated_inc:
        for pred in predictions:
            append_calculated_inc_prediction(pred)

    # Echo some summary statistics - only report on the directly predicted labels
    prediction_type = "mc" if mc_iterations > 1 else "nonmc"
    (lbl_vals, pred_vals, _, _) = get_label_and_prediction_raw_values(labels, predictions,
                                                                      estimator.label_names)
    print("\n-----------------------------------")
    for metric_identifier in ["MAE", "MSE", "r2_score"]:
        metric = metrics.get(metric_identifier)
        metric.update_state(lbl_vals, pred_vals)
        result = metric.result()
        print(f"Total {metric_identifier:>8} ({prediction_type}): {result:.9f}")
    print("-----------------------------------\n")
    return (np.array(labels), np.array(predictions))


def fit_against_formal_test_dataset(
        labels: List[Dict[str, float]],
        input_params: List[Dict[str, float]],
        targets_config: Dict[str, any],
        selected_targets: List[str]=None,
        report_label_names: List[str]=None,
        apply_fit_overrides: bool=True) -> np.ndarray[Dict[str, float]]:
    """
    Will fit members of the formal test dataset, as configured in targets_config,
    based on the sets of input_params passed in returning the corresponding fitted params.

    It's important that input_params, labels and selected_targets (or targets_config keys)
    are all of the same length and in the same order.

    :labels: the list of labels Dicts, for reporting
    :input_params: the list of params Dicts to fit with, one per target
    :targets_config: the full config for all targets
    :selected_targets: list of target ids to fit, or all if empty
    :report_label_names: the names of labels/values that are reported before/after a fit
    :apply_fit_overrides: apply any fit_overrides from each target's config
    :returns: a List of the targets fitted parameter Dicts
    """
    # We report on  labels common to the all labels & input values or those requested
    if report_label_names is None:
        report_label_names = [k for k in labels[0].keys() if k in input_params[0]]

    tw = TextWrapper(100)
    fitted_params = []
    if selected_targets is None or len(selected_targets) == 0:
        selected_targets = list(targets_config.keys())
    assert len(selected_targets) == len(input_params)
    assert len(selected_targets) == len(labels)

    for ix, (target, target_input_params, target_labels) in enumerate(zip(selected_targets,
                                                                          input_params,
                                                                          labels)):
        target_cfg = targets_config[target].copy()
        print(f"\n\nProcessing target {ix+1} of {len(selected_targets)}: {target}\n" + "-"*40)
        print(tw.fill(target_cfg.get("desc", "")) + "\n")

        # The basic lightcurve data read, rectified & extended with delta_mag and delta_mag_err cols
        (lc, sector_count) = datasets.prepare_lightcurve_for_target(target, target_cfg, True)
        pe = lightcurve.to_lc_time(target_cfg["primary_epoch"], lc).value
        period = target_cfg["period"]

        fit_stem = f"model-testing-{re.sub(r'[^\w\d-]', '-', target.lower())}"
        fit_dir = jktebop.get_jktebop_dir()
        for file in fit_dir.glob(f"{fit_stem}.*"):
            file.unlink()
        in_fname = fit_dir / f"{fit_stem}.in"
        dat_fname = fit_dir / f"{fit_stem}.dat"

        print(f"\nWill fit {target} with the following input params")
        predictions_vs_labels_to_table([target_labels], [target_input_params], [target],
                                       report_label_names)

        # published fitting params that may be needed for good fit
        fit_overrides = target_cfg.get("fit_overrides", {}) if apply_fit_overrides else {}
        lrats = fit_overrides.get("lrat", [])

        params = {
            **base_jktebop_task3_params(period, pe, dat_fname.name, fit_stem, target_cfg),
            **target_input_params,
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
        par_fname = fit_dir / f"{fit_stem}.par"
        par_contents = list(jktebop.run_jktebop_task(in_fname, par_fname, stdout_to=sys.stdout))
        fit_params = jktebop.read_fitted_params_from_par_lines(par_contents, report_label_names)

        print(f"\nHave fitted {target} resulting in the following fitted params")
        predictions_vs_labels_to_table([target_labels], [fit_params], [target], report_label_names)
        fitted_params.append(fit_params)
    return np.array(fitted_params)


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


def append_calculated_inc_prediction(predictions: Dict[str, float]):
    """
    Calculate the predicted inc value (in degrees) from the primary
    impact param bP, cosi or sini and append to the predictions.

    :preds: the prediction dictionary for this instance
    """
    def pred_to_ufloat(key: str):
        return ufloat(predictions[key], predictions[f"{key}_sigma"])

    if "inc" in predictions:
        return

    if "bP" in predictions:
        # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
        b = pred_to_ufloat("bP")
        r = pred_to_ufloat("rA_plus_rB") / (1 + pred_to_ufloat("k"))
        esinw = pred_to_ufloat("esinw")
        e = sqrt(pred_to_ufloat("ecosw")**2 + esinw**2)
        inc = degrees(acos(b * r * (1 + esinw) / (1 - e**2)))
    elif "cosi" in predictions:
        inc = degrees(acos(pred_to_ufloat("cosi")))
    elif "sini" in predictions:
        inc = degrees(asin(pred_to_ufloat("sini")))
    else:
        raise KeyError("Did not find inc, bP, cosi or sini in predictions to calc orbital inc.")

    predictions["inc"] = inc.nominal_value
    predictions["inc_sigma"] = inc.std_dev


def predictions_vs_labels_to_csv(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        row_headings: List[str]=None,
        selected_label_names: List[str]=None,
        reverse_scaling: bool=False,
        to: TextIOBase=None):
    """
    Will write a csv of the predicted nominal values vs the labels
    with O-C, MAE and MSE metrics, to the requested output.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :row_headings: the optional heading for each row
    :selected_label_names: a subset of the full list of labels/prediction names to render
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    :to: the output to write the table to. Defaults to stdout.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We output the labels common to the all labels & predictions or those requested
    if selected_label_names is None:
        keys = [k for k in labels[0].keys() if k in predictions[0]]
    else:
        keys = selected_label_names
    num_keys = len(keys)

    if row_headings is None:
        row_headings = (f"{ix:06d}" for ix in range(len(labels)))

    # Extracts the raw values from the label and prediction List[Dict]s
    (raw_labels, pred_noms, _, ocs) \
        = get_label_and_prediction_raw_values(labels, predictions, keys, reverse_scaling)

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


def predictions_vs_labels_to_table(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        block_headings: List[str]=None,
        selected_label_names: List[str]=None,
        reverse_scaling: bool=False,
        comparison_head: str="Label",
        prediction_head: str="Prediction",
        summary_only: bool=False,
        to: TextIOBase=None):
    """
    Will write a text table of the predicted nominal values vs the label values
    with O-C, MAE and MSE metrics, to the requested output.

    :labels: the labels values as a dict of labels per instance
    :predictions: the prediction values as a dict of predictions per instance.
    All the dicts may either be as { "key": val, "key_sigma": err } or { "key":(val, err) }
    :block_headings: the heading for each block of preds-vs-labels
    :selected_label_names: a subset of the full list of labels/prediction names to render
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    :comparison_head: the text of the comparison row headings (10 chars or less)
    :prediction_head: the text of the prediction row headings (10 chars or less)
    :summary_only: omit the body and just report the summary
    :to: the output to write the table to. Defaults to printing.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We output the labels common to the all labels & predictions or those requested
    if selected_label_names is None:
        keys = [k for k in predictions[0].keys() if k in labels[0]]
    else:
        keys = selected_label_names

    # Extracts the raw values from the label and prediction List[Dict]s
    # Expected to error if selected_label_names contains an unknown label/pred name
    (raw_labels, pred_noms, _, ocs) \
        = get_label_and_prediction_raw_values(labels, predictions, keys, reverse_scaling)

    print_it = not to
    if print_it:
        to = StringIO()

    line_length = 13 + (11 * len(keys))-1 + 22
    def horizontal_line(char):
        to.write(char*line_length + "\n")

    def header_block(header):
        horizontal_line("-")
        to.write(f"{header:<10s} | " + " ".join(f"{k:>10s}" for k in keys + ["MAE", "MSE"]) + "\n")

    if summary_only:
        header_block("Summary")
    else:
        if block_headings is None or len(block_headings) == 0:
            block_headings = (f"{n:04d}" for n in range(1, len(pred_noms)+1))

        for block_head, b_comp, b_preds, b_ocs in zip(block_headings, raw_labels, pred_noms, ocs):
            # A sub table for each block/instance with 3 rows; labels|controls, predictions and ocs
            header_block(block_head)
            horizontal_line("-")
            b_ocs = np.concatenate([b_ocs, [np.mean(np.abs(b_ocs)), np.mean(np.power(b_ocs, 2))]])
            for row_head, row_vals in zip([comparison_head, prediction_head, "O-C"],
                                          [b_comp, b_preds, b_ocs]):
                to.write(f"{row_head:<10s} | " + " ".join(f"{v:10.6f}" for v in row_vals))
                to.write("\n")

    # Summary rows for aggregate stats over all of the rows
    horizontal_line("=")
    to.write(f"{'MAE':<10s} | " + " ".join(f"{v:10.6f}" for v in np.mean(np.abs(ocs), 0)) +
                 f" {np.mean(np.abs(ocs)):10.6f}\n")
    to.write(f"{'MSE':<10s} | " + " ".join([f"{v:10.6f}" for v in np.mean(np.power(ocs, 2), 0)]) +
                 " "*11 + f" {np.mean(np.power(ocs, 2)):10.6f}\n")
    if print_it:
        print(to.getvalue())


def get_dataset_labels_and_features(
    dataset_dir: Path,
    label_names: List[str],
    feature_names: List[str],
    mags_bins: int,
    mags_wrap_phase: float,
    scaled_labels: bool=True,
    include_ids: List[str]=None):
    """
    Gets the ids, labels and features of the requested dataset.

    :dataset_dir: the directory within which it lives
    :label_names: the names of the labels to retrieve
    :feature_names: the names of the features to retrieve (in addition to mags)
    :mags_bins: the size of the mags features we require
    :mags_wrap_phase: the phase at which the mags feature is wrapped
    :scaled_labels: if True labels will be scaled
    :include_ids: List of ids to restrict results to, or all if None/empty
    :returns: Tuple[List[ids], List[labels dict], List[features dict]]
    """

    print(f"Looking for the test dataset in '{dataset_dir}'...", end="")
    tfrecord_files = list(dataset_dir.glob("**/*.tfrecord"))
    if len(tfrecord_files) > 0:
        print(f"found {len(tfrecord_files)} dataset file(s).")
    else:
        raise IndexError("No dataset files found")

    ids, ds_labels, ds_features = [], [], []
    # This will yield the records in the order in which they appear in the dataset
    for (targ, lrow, mrow, frow) in datasets.inspect_dataset(tfrecord_files, include_ids,
                                                             scale_labels=scaled_labels):
        ids += [targ]
        ds_labels += [{ ln: lrow[ln] for ln in label_names}]
        ds_features += [{
            "mags": mrow[deb_example.create_mags_key(mags_bins, mags_wrap_phase)], 
            **{ fn: frow[fn] for fn in feature_names if fn not in ["mags"] }}
        ]

    # Need to sort the data in the order of the requested ids (if given)
    if include_ids is not None and len(include_ids) > 0:
        indices = [ids.index(i) for i in include_ids if i in ids]
        ids = [ids[ix] for ix in indices]
        ds_labels = [ds_labels[ix] for ix in indices]
        ds_features = [ds_features[ix] for ix in indices]

    return ids, ds_labels, ds_features


def get_label_and_prediction_raw_values(
        labels: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        selected_label_names: List[str]=None,
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
    if selected_label_names is None:
        selected_label_names = [k for k in predictions[0].keys() if not k.endswith("sigma")]

    def to_noms_and_errs(inputs, keys) -> np.ndarray:
        # There are two format we expect for the List of input values;
        #   - [{ "key": (value, uncertainty) }]
        #   - [{ "key": value, "key_sigma": uncertainty }]
        # In either case we need to separate out the nominal value and error bars
        if isinstance(inputs[0][keys[0]], tuple):
            # we have tuples of (value, uncertainty) for each key
            nominals = np.array([[inp[l][0] for l in keys] for inp in inputs])
            err_bars = np.array([[inp[l][1] for l in keys] for inp in inputs])
        else:
            # We have single value keys and (optional) separate key_sigma uncertainty values
            nominals = np.array([[inp[l] for l in keys] for inp in inputs])
            if f"{keys[0]}_sigma" in inputs[0]:
                err_bars = [[inp.get(f"{l}_sigma", 0) for l in keys] for inp in inputs]
                err_bars = np.array(err_bars)
            else:
                err_bars = np.zeros_like(nominals)
        return nominals, err_bars

    label_values, _ = to_noms_and_errs(labels, selected_label_names)
    pred_noms, pred_errs = to_noms_and_errs(predictions, selected_label_names)

    # Coalesce any None values to zero otherwise we'll get failures below
    # pylint: disable=singleton-comparison
    label_values[label_values == None] = 0.
    pred_noms[pred_noms == None] = 0.
    pred_errs[pred_errs == None] = 0.
    # pylint: enable=singleton-comparison

    # Optionally reverse any scaling of the values to get them in to the scale used by the ML model
    if reverse_scaling:
        scales = [deb_example.labels_and_scales[l] for l in selected_label_names]
        label_values = np.multiply(label_values, scales)
        pred_noms = np.multiply(pred_noms, scales)
        pred_errs = np.multiply(pred_errs, scales)

    # Currently O-C only considers the nominal values (as that's what we use in estimates)
    ocs = np.subtract(label_values, pred_noms)
    return (label_values, pred_noms, pred_errs, ocs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runs formal testing on trained model files.")
    ap.add_argument(dest="model_files", type=Path, nargs="*", help="The model file(s) to test.")
    ap.set_defaults(model_files=[None]) # If None will load the default model under ebop_maven/data
    args = ap.parse_args()

    with open("./config/formal-test-dataset.json", mode="r", encoding="utf8") as tf:
        targets_cfg = json.load(tf)

    exclude_targets = ["V402 Lac", "V456 Cyg"] # Neither are suitable for JKTEBOP fitting
    exclude_targets += ["psi Cen"] # Unstable fit; deviate from pub params and we get gaussj errors
    exclude_targets += ["V963 Cen"] # Similar; JKTEBOP task 8 gets lots of failures & large uncerts
    targets = np.array([target for target in targets_cfg if target not in exclude_targets])
    trn_flags = np.array([targets_cfg.get(t, {}).get("transits", False) for t in targets])

    for file_counter, model_file in enumerate(args.model_files, 1):
        print(f"\nModel file {file_counter} of {len(args.model_files)}: {model_file}\n")

        the_estimator = Estimator(model_file)
        trainset_name = the_estimator.metadata["trainset_name"]
        mags_key = deb_example.create_mags_key(the_estimator.mags_feature_bins,
                                               the_estimator.mags_feature_wrap_phase)
        save_dir = Path(f"./drop/results/{the_estimator.name}/{trainset_name}/{mags_key}")
        save_dir.mkdir(parents=True, exist_ok=True)

        labs, all_preds, all_fits = None, {}, {}
        with redirect_stdout(Tee(open(save_dir / "model_testing.log", "w", encoding="utf8"))):
            print("\n"+fill(f"Testing {the_estimator.name} against\n{', '.join(targets)}", 100))

            # Report on the performance of the model/Estimator predictions vs labels
            for pred_type, iters in [("nonmc", 1), ("mc", 1000)]:
                print(f"\n\nTesting the model's {pred_type} estimates (iters={iters})\n" + "="*80)
                (labs, all_preds[pred_type]) = test_model_against_formal_test_dataset(the_estimator,
                                                                                    iters, targets)

                results_stem = "predictions_vs_labels_" + pred_type # pylint: disable=invalid-name
                fig = plots.plot_predictions_vs_labels(labs, all_preds[pred_type], trn_flags)
                fig.savefig(save_dir / f"{results_stem}.eps")

                with open(save_dir / f"{results_stem}.csv", mode="w", encoding="utf8") as cf:
                    predictions_vs_labels_to_csv(labs, all_preds[pred_type], targets, to=cf)

                with open(save_dir / f"{results_stem}.txt", mode="w", encoding="utf8") as tf:
                    for (heading, mask) in [("All targets", [True]*len(targets)),
                                            ("\n\nTransiting systems only", trn_flags),
                                            ("\n\nNon-transiting systems only", ~trn_flags)]:
                        tf.write(f"\n{heading}\n")
                        if any(mask):
                            predictions_vs_labels_to_table(labs[mask], all_preds[pred_type][mask],
                                                           targets[mask], to=tf)

            # These are the key/values which are set for a JKTEBOP fit. If comparing
            # models/prediction values it's these six which ultimately matter.
            fit_keys = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc"]

            # Report using the predictions as input to fitting the formal-test-dataset with JKTEBOP.
            # First we add control "preds" from labels, so we can use them to create control fits.
            all_preds["control"] = copy.deepcopy(labs)
            for pred_type in ["control", "mc", "nonmc"]:
                print(f"\n\nTesting JKTEBOP fitting based on {pred_type} input values\n" + "="*80)
                all_fits[pred_type] = fit_against_formal_test_dataset(labs, all_preds[pred_type],
                                                                      targets_cfg, targets,
                                                                      fit_keys, True)

                # Now summarize how well the fits compare with labels and the control fit
                for comp, fits, comp_type, comp_heading in [
                            (labs, all_fits[pred_type], "labels", "Label"),
                            (all_fits["control"], all_fits[pred_type], "control_fit", "Control")]:
                    if comp is fits:
                        break # No point comparing the controls with the controls

                    results_stem = f"fitted_params_from_{pred_type}_vs_{comp_type}" # pylint: disable=invalid-name
                    fig = plots.plot_predictions_vs_labels(comp, fits, trn_flags, fit_keys,
                                                           xlabel_prefix=comp_heading.lower(),
                                                           ylabel_prefix="fitted")
                    fig.savefig(save_dir / f"{results_stem}.eps", dpi=300)

                    with open(save_dir / f"{results_stem}.txt", "w", encoding="utf8") as of:
                        for (heading, mask) in [("All targets", [True]*len(targets)),
                                                ("\n\nTransiting systems only", trn_flags),
                                                ("\n\nNon-transiting systems only", ~trn_flags)]:
                            of.write(f"\n{heading}\n")
                            if any(mask):
                                predictions_vs_labels_to_table(comp[mask], fits[mask],
                                                               targets[mask], fit_keys,
                                                               comparison_head=comp_heading,
                                                               prediction_head="Fitted", to=of)
