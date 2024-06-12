#!/usr/bin/env python3
"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
# pylint: disable=too-many-arguments, too-many-locals, no-member, import-error, invalid-name
from typing import Union, List, Dict, Tuple
from io import TextIOBase, StringIO
import sys
from pathlib import Path
import json
import re
from contextlib import redirect_stdout
from textwrap import fill
import copy
import argparse

import matplotlib.pylab as plt

from uncertainties import ufloat
from uncertainties.umath import sqrt, acos, asin, degrees # pylint: disable=no-name-in-module

import astropy.units as u
import numpy as np
from keras import Model

from ebop_maven.libs.tee import Tee
from ebop_maven.libs import jktebop, stellar, limb_darkening
from ebop_maven.estimator import Estimator
from ebop_maven import datasets, deb_example, pipeline, plotting
import plots

SYNTHETIC_MIST_TEST_DS_DIR = Path("./datasets/synthetic-mist-tess-dataset/")
FORMAL_TEST_DATASET_DIR = Path("./datasets/formal-test-dataset/")

def evaluate_model_against_dataset(
        estimator: Union[Model, Estimator],
        mc_iterations: int=1,
        include_ids: List[str]=None,
        test_dataset_dir: Path=FORMAL_TEST_DATASET_DIR,
        report_dir: Path=None,
        scaled: bool=False) \
            -> Tuple[np.ndarray, np.ndarray]:
    """
    Will evaluate the indicated model file or Estimator against the contents of the
    chosen test dataset.

    :estimator: the Estimator or estimator model to use to make predictions
    :mc_iterations: the number of MC Dropout iterations
    :include_ids: list of target ids to predict, or all if not set
    :test_dataset_dir: the location of the formal test dataset
    :report_dir: optional directory into which to save reports, reports not saved if this is None
    :scaled: whether labels and predictions are scaled (raw model predictions) or not
    :returns: a tuple of (NDArray, NDArray) containing
    label valuess (#insts, #labels) and predictions (#insts, #labels & #sigmas)
    """
    # Create our Estimator. It will tell us what its inputs should look like
    # and which labels it (& the underlying model) can predict.
    if isinstance(estimator, (Model, Path)):
        estimator = Estimator(estimator)
    prediction_type = "mc" if mc_iterations > 1 else "nonmc"

    # Be sure to retrieve the inc label as it's needed to work out which systems have transits
    extended_label_names = estimator.label_names
    if "inc" not in extended_label_names:
        extended_label_names += ["inc"]

    print(f"Looking for the test dataset in '{test_dataset_dir}'...", end="")
    tfrecord_files = sorted(test_dataset_dir.glob("**/*.tfrecord"))
    print(f"found {len(tfrecord_files)} file(s).")

    # Set up augmentations to perturb synthetic data as we do for the full training pipeline
    noise_stddev, roll_max = 0, 0
    if test_dataset_dir != FORMAL_TEST_DATASET_DIR:
        noise_stddev = 0.005
        roll_max = 36

    ids, mags_vals, feat_vals, lbl_vals = deb_example.read_dataset(tfrecord_files,
                                                                estimator.mags_feature_bins,
                                                                estimator.mags_feature_wrap_phase,
                                                                estimator.input_feature_names,
                                                                extended_label_names,
                                                                include_ids,
                                                                scaled,
                                                                noise_stddev,
                                                                roll_max)

    if include_ids is not None:
        assert len(include_ids) == len(ids)

    # Make our predictions which will be returned in the shape (#insts, #labels, #iterations)
    # then make a set of noms and 1-sigmas: shape (#insts, #nominal & #1-sigmas)
    print(f"The Estimator is making predictions on the {len(ids)} test instances",
          f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
    pred_vals = estimator.predict_raw(mags_vals, feat_vals, mc_iterations, unscale=not scaled)
    pred_vals = estimator.means_and_stddevs_from_predictions(pred_vals, label_axis=1)
    lbl_vals = estimator.means_and_stddevs_from_predictions(lbl_vals, label_axis=1)

    # A bit costly to create arrays of dicts, but we can reuse of existing summary functionality
    # to give us MAE and MSE stats across each label in addition to the whole set of predictions.
    pred_dicts = np.array([dict(zip(estimator.prediction_names, rvals)) for rvals in pred_vals])
    lbl_dicts = np.array([dict(zip(estimator.prediction_names, rvals)) for rvals in lbl_vals])
    transit_args = ["rA_plus_rB", "k", "inc", "ecosw", "esinw"]
    tflags = will_transit(*[lbl_vals[:, extended_label_names.index(k)] for k in transit_args])
    for (subset, mask) in [("",                 [True]*len(lbl_dicts)),
                           (" transiting",      tflags),
                           (" non-transiting",  ~tflags)]:
        if any(mask):
            print(f"\nMetrics for {sum(mask)}{subset} system(s).")
            preds_vs_labels_dicts_to_table(pred_dicts[mask], lbl_dicts[mask], summary_only=True,
                                           selected_label_names=estimator.label_names)

    if report_dir:
        # Produce a box plot of the prediction residual for each instance vs label/value
        # Output to pdf, which looks better than eps, as it supports transparency/alpha.
        num_est_labels = len(estimator.label_names)
        resids_by_label = (pred_vals[:, :num_est_labels] - lbl_vals[:, :num_est_labels]).transpose()
        fig, axes = plt.subplots(figsize=(6, 4), tight_layout=True)
        fliers = "formal" in test_dataset_dir.name
        plotting.plot_prediction_distributions_on_axes(axes, resids_by_label, estimator.label_names,
                                        violin_plot=False, show_fliers=fliers, ylabel="Residual")
        fig.savefig(report_dir / f"predictions-dist-{test_dataset_dir.name}-{prediction_type}.pdf")
    return lbl_vals, pred_vals


def fit_against_formal_test_dataset(
        estimator: Union[Model, Estimator],
        targets_config: Dict[str, any],
        include_ids: List[str]=None,
        mc_iterations: int=1,
        apply_fit_overrides: bool=True,
        do_control_fit: bool=False,
        comparison_dicts: np.ndarray[Dict[str, float]]=None,
        report_dir: Path=None) -> np.ndarray[Dict[str, float]]:
    """
    Will fit members of the formal test dataset, as configured in targets_config,
    based on the sets of input_params passed in returning the corresponding fitted params.

    It's important that input_params, labels and selected_targets (or targets_config keys)
    are all of the same length and in the same order.

    :estimator: the Estimator or estimator model to use to make predictions
    :targets_config: the full config for all targets
    :selected_targets: list of target ids to fit, or all if empty
    :mc_iterations: the number of MC iterations to use when making predictions
    :apply_fit_overrides: apply any fit_overrides from each target's config
    :do_control_fit: when True labels, rather than predictions, will be the input params for fitting
    :comparison_dicts: optional List[Dict] to compare the fitting results to in addition to labels
    :report_dir: optional directory into which to save reports, reports not saved if this is None
    :returns: a List of the targets' fitted parameter Dicts
    """
    # pylint: disable=too-many-statements, too-many-branches
    fit_dir = jktebop.get_jktebop_dir()
    if isinstance(estimator, (Model, Path)):
        estimator = Estimator(estimator)

    prediction_type = "control" if do_control_fit else "mc" if mc_iterations > 1 else "nonmc"

    # To clarify: the estimator publishes a list of what it can predict via its label_names attrib
    # The fit_names can differ; they are those values required for JKTEBOP fitting and reporting
    # super_names is the set of both and is used, for example, to get the superset of label values
    fit_names = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc"]
    super_names = estimator.label_names + [n for n in fit_names if n not in estimator.label_names]
    super_names_and_errs = super_names + [f"{n}_sigma" for n in super_names]

    if include_ids is None or len(include_ids) == 0:
        include_ids = list(targets_config.keys())
    trans_flags = np.array([targets_config.get(t,{}).get("transits", False) for t in include_ids])

    print(f"\nLooking for the test dataset in '{FORMAL_TEST_DATASET_DIR}'.")
    tfrecord_files = sorted(FORMAL_TEST_DATASET_DIR.glob("**/*.tfrecord"))
    all_targs, all_mags_val, all_feat_vals, all_lbl_vals = deb_example.read_dataset(tfrecord_files,
                                                                estimator.mags_feature_bins,
                                                                estimator.mags_feature_wrap_phase,
                                                                estimator.input_feature_names,
                                                                super_names,
                                                                include_ids)

    # Get the labels into an NDArray[Dicts] for ease of reporting & possible use as control fit
    all_lbl_vals = estimator.means_and_stddevs_from_predictions(all_lbl_vals, label_axis=1)
    all_lbl_dicts = np.array([dict(zip(super_names_and_errs, v)) for v in all_lbl_vals])

    # The Estimator predictions will be in the shape (#insts, #labels, #iterations)
    # Then get the predictions into NDArray(Dicts), which is required for jktebop and
    # reporting, via sets of means and 1-sigmas: shape (#insts, #nominal & #1-sigmas)
    if do_control_fit:
        print("\nControl fits will use label values as 'predictions' and fitting inputs.")
        all_pred_dicts = copy.deepcopy(all_lbl_dicts)
    else:
        print(f"\nThe Estimator will make predictions on {len(all_targs)} formal test instance(s)",
              f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
        all_pred_vals = estimator.predict_raw(all_mags_val, all_feat_vals, mc_iterations)
        all_pred_vals = estimator.means_and_stddevs_from_predictions(all_pred_vals, label_axis=1)
        all_pred_dicts = np.array([dict(zip(estimator.prediction_names, v)) for v in all_pred_vals])
        if "inc" not in estimator.label_names:
            append_calculated_inc_predictions(all_pred_dicts)

    # Finally, we have everything in place to fit our targets with JKTEBOP and report on the results
    fitted_param_dicts = []
    for ix, (targ, pred_dict, lbl_dict) in enumerate(zip(all_targs, all_pred_dicts, all_lbl_dicts)):
        print(f"\n\nProcessing target {ix + 1} of {len(all_targs)}: {targ}\n" + "-"*40)
        targ_config = targets_config[targ].copy()
        print(fill(targ_config.get("desc", "")) + "\n")

        # The basic lightcurve data read, rectified & extended with delta_mag and delta_mag_err cols
        (lc, sector_count) = datasets.prepare_lightcurve_for_target(targ, targ_config, True)
        pe = pipeline.to_lc_time(targ_config["primary_epoch"], lc).value
        period = targ_config["period"]

        fit_stem = f"model-testing-{re.sub(r'[^\w\d-]', '-', targ.lower())}"
        for file in fit_dir.glob(f"{fit_stem}.*"):
            file.unlink()
        in_fname = fit_dir / f"{fit_stem}.in"
        dat_fname = fit_dir / f"{fit_stem}.dat"

        print(f"\nWill fit {targ} with the following input params")
        preds_vs_labels_dicts_to_table([pred_dict], [lbl_dict], [targ], fit_names)

        # published fitting params that may be needed for reliable fit
        fit_overrides = targ_config.get("fit_overrides", {}) if apply_fit_overrides else {}
        lrats = fit_overrides.get("lrat", [])

        params = {
            **base_jktebop_task3_params(period, pe, dat_fname.name, fit_stem, targ_config),
            **pred_dict,
            **fit_overrides,
        }

        # Add scale-factor poly fitting, chi^2 adjustment (to 1.0) or light-ratio instructions
        segments = pipeline.find_lightcurve_segments(lc, 0.5, return_times=True)
        append_lines = jktebop.build_poly_instructions(segments, "sf", 1)
        append_lines += ["", "chif", ""] + [ f"lrat {l}" for l in lrats ]

        jktebop.write_in_file(in_fname, task=3, append_lines=append_lines, **params)
        jktebop.write_light_curve_to_dat_file(
                    lc, dat_fname, column_formats=[lambda t: f"{t.value:.6f}", "%.6f", "%.6f"])

        # Don't consume the output files so they're available if we need any diagnostics.
        # Read superset of fit and label values as these data are needed for reports.
        print(f"\nFitting {targ} (with {sector_count} sector(s) of data) using JKTEBOP task 3...")
        par_fname = fit_dir / f"{fit_stem}.par"
        par_contents = list(jktebop.run_jktebop_task(in_fname, par_fname, stdout_to=sys.stdout))
        fit_params_dict = jktebop.read_fitted_params_from_par_lines(par_contents, super_names)
        fitted_param_dicts.append(fit_params_dict)

        print(f"\nHave fitted {targ} resulting in the following fitted params")
        preds_vs_labels_dicts_to_table([fit_params_dict], [lbl_dict], [targ], fit_names,
                                       prediction_head="Fitted")

    # Save reports on how the predictions and fitting has gone over all of the selected targets
    fitted_param_dicts = np.array(fitted_param_dicts)
    if report_dir:
        comparison_reports = [("labels", "label", all_lbl_dicts)]
        if comparison_dicts is not None:
            comparison_reports += [("control", "control", comparison_dicts)]
        sub_reports = [
            ("All targets (model labels)",          [True]*len(all_targs),  estimator.label_names),
            ("\nTransiting systems only",           trans_flags,            estimator.label_names),
            ("\nNon-transiting systems only",       ~trans_flags,           estimator.label_names),
            ("\n\n\nAll targets (fitting params)",  [True]*len(all_targs),  fit_names),
            ("\nTransiting systems only",           trans_flags,            fit_names),
            ("\nNon-transiting systems only",       ~trans_flags,           fit_names)]

        for comp_type, comp_head, comp_dicts in comparison_reports:
            if not do_control_fit: # Control == fit from labels not preds; no point producing these
                preds_stem = f"predictions-{prediction_type}-vs-{comp_type}"
                for (source, names) in [("model", estimator.label_names), ("fitting", fit_names)]:
                    names = [n for n in names if n not in ["ecosw", "esinw"]] + ["ecosw", "esinw"]
                    plots.plot_predictions_vs_labels(comp_dicts, all_pred_dicts, trans_flags, names,
                        xlabel_prefix=comp_head).savefig(report_dir / f"{preds_stem}-{source}.eps")

                # with open(report_dir / f"{preds_stem}.csv", mode="w", encoding="utf8") as csvf:
                #     predictions_vs_labels_to_csv(
                #                         comp_dicts, all_pred_dicts, all_targs, fit_names, to=csvf)

                with open(report_dir / f"{preds_stem}.txt", mode="w", encoding="utf8") as txtf:
                    for (sub_head, mask, rep_names) in sub_reports:
                        if any(mask):
                            preds_vs_labels_dicts_to_table(all_pred_dicts[mask], comp_dicts[mask],
                                                all_targs[mask], rep_names, title=sub_head, to=txtf)

            results_stem = f"fitted-params-from-{prediction_type}-vs-{comp_type}"
            names = [n for n in fit_names if n not in ["ecosw", "esinw"]] + ["ecosw", "esinw"]
            plots.plot_predictions_vs_labels(comp_dicts, fitted_param_dicts, trans_flags, names,
                                             xlabel_prefix=comp_head, ylabel_prefix="fitted") \
                                                    .savefig(report_dir / f"{results_stem}.eps")

            with open(report_dir / f"{results_stem}.txt", "w", encoding="utf8") as txtf:
                for (sub_head, mask, rep_names) in sub_reports:
                    if any(mask):
                        preds_vs_labels_dicts_to_table(fitted_param_dicts[mask], comp_dicts[mask],
                                                       all_targs[mask], rep_names, title=sub_head,
                                                       comparison_head=comp_head.capitalize(),
                                                       prediction_head="Fitted", to=txtf)
    return fitted_param_dicts


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


def append_calculated_inc_predictions(predictions: np.ndarray[Dict[str, float]]):
    """
    Calculate the predicted inc value (in degrees) from the primary
    impact param bP, cosi or sini and append to the predictions.

    :preds: the array of prediction dictionaries to update
    """
    def to_ufloat(preds: dict, key: str):
        return ufloat(preds[key], preds.get(f"{key}_sigma", 0))

    preds_iter = [predictions] if isinstance(predictions, Dict) else predictions
    for ix, preds in enumerate(preds_iter):
        if "inc" not in preds:
            if "bP" in preds:
                # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
                b = to_ufloat(preds, "bP")
                r = to_ufloat(preds, "rA_plus_rB") / (1 + to_ufloat(preds, "k"))
                esinw = to_ufloat(preds, "esinw")
                e = sqrt(to_ufloat(preds, "ecosw")**2 + esinw**2)
                inc = degrees(acos(b * r * (1 + esinw) / (1 - e**2)))
            elif "cosi" in preds:
                inc = degrees(acos(to_ufloat(preds, "cosi")))
            elif "sini" in preds:
                inc = degrees(asin(to_ufloat(preds, "sini")))
            else:
                raise KeyError(f"Missing inc, bP, cosi or sini in predictions[{ix}] to calc inc.")

            preds.update(inc=inc.nominal_value, inc_sigma=inc.std_dev)


def will_transit(rA_plus_rB: np.ndarray[float],
                 k: np.ndarray[float],
                 inc: np.ndarray[float],
                 ecosw: np.ndarray[float],
                 esinw: np.ndarray[float]) \
                        -> np.ndarray[bool]:
    """
    From the values given over 1 or more systems, this will indicate which will
    exhibit at least one type of transit.

    :rA_plus_rB: the systems' sum of the radii
    :k: the systems' ratio of the radii
    :inc: the orbital inclinations in degrees
    :ecosw: the e*cos(omega) Poincare elements
    :esinw: the e*sin(omega) Poincare elements
    :returns: flags indicating which systems will transit
    """
    cosi = np.cos(np.deg2rad(inc))
    e = np.sqrt(np.add(np.square(ecosw), np.square(esinw)))

    # For some systems rB > rA which we handle by using abs to get the difference
    rA = np.divide(rA_plus_rB, np.add(1, k))
    rA_diff_rB = np.abs(np.subtract(rA, np.subtract(rA_plus_rB, rA)))

    # As we're looking for total eclipses the condition is
    # Primary:      cos(inc) < rA-rB * (1+esinw) / (1-e^2)
    # Secondary:    cos(inc) < rA-rB * (1-esinw) / (1-e^2)
    pt1 = np.divide(rA_diff_rB, np.subtract(1, np.square(e)))
    primary_trans = cosi < np.multiply(pt1, np.add(1, esinw))
    secondary_trans = cosi < np.multiply(pt1, np.subtract(1, esinw))
    return np.bitwise_or(primary_trans, secondary_trans)


def preds_vs_labels_dicts_to_csv(
        predictions: np.ndarray[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        labels: np.ndarray[Dict[str, float]],
        row_headings: np.ndarray[str]=None,
        selected_label_names: np.ndarray[str]=None,
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


def preds_vs_labels_dicts_to_table(
        predictions: np.ndarray[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        labels: np.ndarray[Dict[str, float]],
        block_headings: np.ndarray[str]=None,
        selected_label_names: np.ndarray[str]=None,
        reverse_scaling: bool=False,
        comparison_head: str="Label",
        prediction_head: str="Prediction",
        title: str=None,
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
    :title: optional title text to write above the table
    :summary_only: omit the body and just report the summary
    :to: the output to write the table to. Defaults to printing.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # We output the labels common to the all labels & predictions or those requested
    if selected_label_names is None:
        keys = [k for k in labels[0].keys() if k in predictions[0]]
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

    if title:
        to.write(f"{title}\n")

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
        formal_targs_cfg = json.load(tf)
    formal_targs = np.array([t for t, c in formal_targs_cfg.items() if not c.get("exclude", False)])
    #formal_targs = np.array(["V436 Per", "CM Dra"])

    for file_counter, model_file in enumerate(args.model_files, 1):
        print(f"\nModel file {file_counter} of {len(args.model_files)}: {model_file}\n")

        # Set up the estimator and the reporting directory for this model
        the_estimator = Estimator(model_file)
        trainset_name = the_estimator.metadata["trainset_name"]
        mags_key = deb_example.create_mags_key(the_estimator.mags_feature_bins,
                                               the_estimator.mags_feature_wrap_phase)
        if model_file is None or model_file.parent.name == "estimator": # published with ebop_maven
            result_dir = Path("./drop/training/published/testing")
        else:
            result_dir = model_file.parent / "testing"
        result_dir.mkdir(parents=True, exist_ok=True)

        labs, all_preds = None, {}
        with redirect_stdout(Tee(open(result_dir / "model_testing.log", "w", encoding="utf8"))):
            # Report on the performance of the model/Estimator predictions vs labels
            for pred_type, iters, dataset_dir, targs in [
                    ("nonmc",   1,      FORMAL_TEST_DATASET_DIR,    formal_targs),
                    ("mc",      1000,   FORMAL_TEST_DATASET_DIR,    formal_targs),
                    ("nonmc",   1,      SYNTHETIC_MIST_TEST_DS_DIR, None),
                    # TODO: probably need a batched dataset for this as it's a memory hog
                    # ("mc",      1000,   SYNTHETIC_MIST_TEST_DS_DIR, None),
            ]:
                print(f"\nEvaluating the model's {pred_type} estimates (iters={iters})",
                      f"on {dataset_dir.name}\n" + "="*80)
                evaluate_model_against_dataset(the_estimator, iters, targs, dataset_dir, result_dir)

            # Report on fitting the formal-test-dataset based on estimator predictions. First run
            # through actually uses labels as the fit inputs to give us a set of control fit results
            control_fit_dicts = None # To be set on the first, control fit run
            for (pred_type, is_control_fit, iterations) in [
                ("control",     True,       0),
                ("nonmc",       False,      1),
                ("mc",          False,      1000),
            ]:
                print(f"\nTesting JKTEBOP fitting of {pred_type} input values\n" + "="*80)
                compare_dicts = None if is_control_fit else control_fit_dicts
                fit_results_dicts = fit_against_formal_test_dataset(the_estimator,
                                                                    formal_targs_cfg,
                                                                    formal_targs,
                                                                    iterations,
                                                                    True,
                                                                    is_control_fit,
                                                                    compare_dicts,
                                                                    result_dir)
                if is_control_fit:
                    control_fit_dicts = fit_results_dicts
