#!/usr/bin/env python3
"""
Formal testing of the regression TF Model trained by train_*_estimator.py
"""
# pylint: disable=too-many-arguments, too-many-locals, no-member, import-error, invalid-name
from typing import Union, List, Dict
from io import TextIOBase, StringIO
import inspect
import sys
from pathlib import Path
import re
from contextlib import redirect_stdout
from textwrap import fill
import copy
import argparse

import matplotlib.pylab as plt

from uncertainties import ufloat, UFloat, unumpy
from uncertainties.umath import acos, asin, degrees # pylint: disable=no-name-in-module

import astropy.units as u
import numpy as np
from keras import Model

from ebop_maven.libs.tee import Tee
from ebop_maven.libs import jktebop, stellar, limb_darkening
from ebop_maven.estimator import Estimator
from ebop_maven import deb_example, pipeline

from traininglib import formal_testing, plots


SYNTHETIC_MIST_TEST_DS_DIR = Path("./datasets/synthetic-mist-tess-dataset/")
FORMAL_TEST_DATASET_DIR = Path("./datasets/formal-test-dataset/")

def evaluate_model_against_dataset(estimator: Union[Model, Estimator],
                                   mc_iterations: int=1,
                                   include_ids: List[str]=None,
                                   test_dataset_dir: Path=FORMAL_TEST_DATASET_DIR,
                                   report_dir: Path=None,
                                   scaled: bool=False):
    """
    Will evaluate the indicated model file or Estimator against the contents of the
    chosen test dataset.

    :estimator: the Estimator or estimator model to use to make predictions
    :mc_iterations: the number of MC Dropout iterations
    :include_ids: list of target ids to predict, or all if not set
    :test_dataset_dir: the location of the formal test dataset
    :report_dir: optional directory into which to save reports, reports not saved if this is None
    :scaled: whether labels and predictions are scaled (raw model predictions) or not
    """
    # Create our Estimator. It will tell us what its inputs should look like
    # and which labels it (& the underlying model) can predict.
    if isinstance(estimator, (Model, Path)):
        estimator = Estimator(estimator)
    mc_type = "mc" if mc_iterations > 1 else "nonmc"

    # Be sure to retrieve the inc label as it's needed to work out which systems have transits
    ext_label_names = estimator.label_names
    if "inc" not in ext_label_names:
        ext_label_names += ["inc"]

    print(f"Looking for the test dataset in '{test_dataset_dir}'...", end="")
    ds_name = test_dataset_dir.name
    tfrecord_files = sorted(test_dataset_dir.glob("**/*.tfrecord"))
    print(f"found {len(tfrecord_files)} file(s).")
    ids, mags_vals, feat_vals, lbl_vals = deb_example.read_dataset(tfrecord_files,
                                                                estimator.mags_feature_bins,
                                                                estimator.mags_feature_wrap_phase,
                                                                estimator.extra_feature_names,
                                                                ext_label_names,
                                                                include_ids,
                                                                scaled)

    if include_ids is not None:
        assert len(include_ids) == len(ids)

    # Make predictions, returned as a structured array of shape (#insts, #labels) and dtype=UFloat
    print(f"The Estimator is making predictions on the {len(ids)} test instances",
          f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
    pred_vals = estimator.predict(mags_vals, feat_vals, mc_iterations, unscale=not scaled)

    # Work out which are the transiting systems so we can break down the reporting
    pnames = list(inspect.signature(will_transit).parameters)
    argvs = [lbl_vals[p] / (deb_example.labels_and_scales[p] if scaled else 1) for p in pnames]
    tflags = will_transit(*argvs)

    # Now report on the quality of the predictions, for which we will need the residuals
    resids = calculate_residuals(pred_vals, lbl_vals, estimator.label_names)
    pnames = [n for n in pred_vals.dtype.names if n not in ["ecosw", "esinw"]] + ["ecosw", "esinw"]
    for (subset, tmask) in [("",                 [True]*lbl_vals.shape[0]),
                            (" transiting",      tflags),
                            (" non-transiting",  ~tflags)]:
        sub_suffix = subset.replace(' ','-')
        if any(tmask):
            print(f"\nMetrics for {sum(tmask)}{subset} system(s).")
            predictions_vs_labels_to_table(pred_vals[tmask], lbl_vals[tmask], summary_only=True,
                                           selected_param_names=estimator.label_names)

            # These plot to pdf, which looks better than eps, as it supports transparency/alpha.
            if report_dir and (not subset or "formal" not in ds_name):
                # For formal-test-dataset plot only the whole set as the size is too low to split.
                fig = plots.plot_prediction_boxplot(resids[tmask], show_fliers="formal" in ds_name,
                                                    ylabel="Residual")
                fig.savefig(report_dir / f"predictions-{mc_type}-box-{ds_name}{sub_suffix}.pdf")

                rep_flags = tflags if len(subset) == 0 else tflags[tmask]
                fig = plots.plot_predictions_vs_labels(pred_vals[tmask], lbl_vals[tmask],
                                                       rep_flags, pnames, show_errorbars=False)
                fig.savefig(report_dir/f"predictions-{mc_type}-vs-labels-{ds_name}{sub_suffix}.pdf")

            if report_dir and "formal" not in ds_name:
                # Break down the predictions into bins then plot the MAE against mean label to give
                # an indication of how the accuracy of the predictions vary over the label ranges.
                fig = plots.plot_binned_mae_vs_labels(resids[tmask], lbl_vals[tmask],
                                                      ["esinw", "ecosw"], xlim=(-.75, .75))
                fig.savefig(report_dir / f"binned-mae-poincare-{mc_type}-{ds_name}{sub_suffix}.pdf")

                fig = plots.plot_binned_mae_vs_labels(resids[tmask], lbl_vals[tmask],
                                                      ["k", "J", "bP"])
                fig.savefig(report_dir / f"binned-mae-kjbp-{mc_type}-{ds_name}{sub_suffix}.pdf")
            plt.close()


def fit_against_formal_test_dataset(estimator: Union[Model, Estimator],
                                    targets_config: Dict[str, any],
                                    include_ids: List[str]=None,
                                    mc_iterations: int=1,
                                    apply_fit_overrides: bool=True,
                                    do_control_fit: bool=False,
                                    comparison_vals: np.ndarray[UFloat]=None,
                                    report_dir: Path=None) -> np.ndarray[UFloat]:
    """
    Will fit members of the formal test dataset, as configured in targets_config,
    based on the sets of input_params passed in returning the corresponding fitted params.

    :estimator: the Estimator or estimator model to use to make predictions
    :targets_config: the full config for all targets
    :selected_targets: list of target ids to fit, or all if empty
    :mc_iterations: the number of MC iterations to use when making predictions
    :apply_fit_overrides: apply any fit_overrides from each target's config
    :do_control_fit: when True labels, rather than predictions, will be the input params for fitting
    :comparison_vals: optional recarray[UFloat] to compare to fitting results, in addition to labels
    :report_dir: optional directory into which to save reports, reports not saved if this is None
    :returns: a recarray[UFloat] containing the resulting fitted parameters for each target
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

    if include_ids is None or len(include_ids) == 0:
        include_ids = list(targets_config.keys())
    trans_flags = np.array([targets_config.get(t,{}).get("transits", False) for t in include_ids])

    print(f"\nLooking for the test dataset in '{FORMAL_TEST_DATASET_DIR}'.")
    tfrecord_files = sorted(FORMAL_TEST_DATASET_DIR.glob("**/*.tfrecord"))
    targs, mags_vals, feat_vals, lbl_vals = deb_example.read_dataset(tfrecord_files,
                                                                estimator.mags_feature_bins,
                                                                estimator.mags_feature_wrap_phase,
                                                                estimator.extra_feature_names,
                                                                super_names,
                                                                include_ids)

    # Make predictions, returned as a structured array of shape (#insts, #labels) and dtype=UFloat
    if do_control_fit:
        print("\nControl fits will use label values as 'predictions' and fitting inputs.")
        pred_vals = copy.deepcopy(lbl_vals)
    else:
        print(f"\nThe Estimator will make predictions on {len(targs)} formal test instance(s)",
              f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
        pred_vals = estimator.predict(mags_vals, feat_vals, mc_iterations)
        if "inc" not in pred_vals.dtype.names:
            pred_vals = append_calculated_inc_predictions(pred_vals)

    # Pre-allocate a structured array to hold the params which are output from each JKTEBOP fitting
    fitted_vals = np.empty(shape=(len(targs), ),
                           dtype=[(name, np.dtype(UFloat.dtype)) for name in super_names])

    # Finally, we have everything in place to fit our targets and report on the results
    for ix, targ in enumerate(targs):
        print(f"\n\nProcessing target {ix + 1} of {len(targs)}: {targ}\n" + "-"*40)
        targ_config = targets_config[targ].copy()
        print(fill(targ_config.get("desc", "")) + "\n")

        # The basic lightcurve data read, rectified & extended with delta_mag and delta_mag_err cols
        (lc, sector_count) = formal_testing.prepare_lightcurve_for_target(targ, targ_config, True)
        pe = pipeline.to_lc_time(targ_config["primary_epoch"], lc).value
        period = targ_config["period"]

        fit_stem = "model-testing-" + re.sub(r"[^\w\d-]", "-", targ.lower())
        for file in fit_dir.glob(f"{fit_stem}.*"):
            file.unlink()
        in_fname = fit_dir / f"{fit_stem}.in"
        dat_fname = fit_dir / f"{fit_stem}.dat"

        print(f"\nWill fit {targ} with these input params from {prediction_type} predictions")
        predictions_vs_labels_to_table(pred_vals[ix], lbl_vals[ix], [targ], fit_names)

        # published fitting params that may be needed for reliable fit
        fit_overrides = targ_config.get("fit_overrides", {}) if apply_fit_overrides else {}
        lrats = fit_overrides.get("lrat", [])

        params = {
            **base_jktebop_fit_params(3, period, pe, dat_fname.name, fit_stem, targ_config),
            **{ n: get_nom(pred_vals[ix][n]) for n in pred_vals.dtype.names },  # predictions
            **fit_overrides,                                                    # overrides
        }

        # Add scale-factor poly fitting, chi^2 adjustment (to 1.0) and any light-ratio instructions
        segments = pipeline.find_lightcurve_segments(lc, 0.5, return_times=True)
        append_lines = jktebop.build_poly_instructions(segments, "sf", 1)
        append_lines += ["", "chif", ""] + [ f"lrat {l}" for l in lrats ]

        jktebop.write_in_file(in_fname, append_lines=append_lines, **params)
        jktebop.write_light_curve_to_dat_file(
                    lc, dat_fname, column_formats=[lambda t: f"{t.value:.6f}", "%.6f", "%.6f"])

        # Don't consume the output files so they're available if we need any diagnostics.
        # Read superset of fit and label values as these data are needed for reports.
        print(f"\nFitting {targ} (with {sector_count} sector(s) of data) using JKTEBOP task 3...")
        par_fname = fit_dir / f"{fit_stem}.par"
        par_contents = list(jktebop.run_jktebop_task(in_fname, par_fname, stdout_to=sys.stdout))
        fit_params_dict = jktebop.read_fitted_params_from_par_lines(par_contents, super_names)
        fitted_vals[ix] = tuple(ufloat(value[0], value[1]) for _, value in fit_params_dict.items())

        print(f"\nHave fitted {targ} resulting in the following fitted params")
        predictions_vs_labels_to_table(fitted_vals[ix], lbl_vals[ix], [targ], fit_names,
                                       prediction_head="Fitted")

    # Save reports on how the predictions and fitting has gone over all of the selected targets
    if report_dir:
        comparison_type = [("labels", "label", lbl_vals)] # type, heading, values
        if comparison_vals is not None:
            comparison_type += [("control", "control", comparison_vals)]
        sub_reports = [
            ("All targets (model labels)",          [True]*len(targs),      estimator.label_names),
            ("\nTransiting systems only",           trans_flags,            estimator.label_names),
            ("\nNon-transiting systems only",       ~trans_flags,           estimator.label_names),
            ("\n\n\nAll targets (fitting params)",  [True]*len(targs),      fit_names),
            ("\nTransiting systems only",           trans_flags,            fit_names),
            ("\nNon-transiting systems only",       ~trans_flags,           fit_names)]

        for comp_type, comp_head, comp_vals in comparison_type:
            # Summarize this set of predictions as plots-vs-label|control and in text table
            # Control == fit from labels not preds, so no point producing these
            if not do_control_fit:
                preds_stem = f"predictions-{prediction_type}-vs-{comp_type}"
                for (source, names) in [("model", estimator.label_names), ("fitting", fit_names)]:
                    names = [n for n in names if n not in ["ecosw", "esinw"]] + ["ecosw", "esinw"]
                    fig = plots.plot_predictions_vs_labels(pred_vals, comp_vals, trans_flags,
                                                           names, xlabel_prefix=comp_head)
                    fig.savefig(report_dir / f"{preds_stem}-{source}.eps")

                with open(report_dir / f"{preds_stem}.txt", mode="w", encoding="utf8") as txtf:
                    for (sub_head, mask, rep_names) in sub_reports:
                        if any(mask):
                            predictions_vs_labels_to_table(pred_vals[mask], comp_vals[mask],
                                                    targs[mask], rep_names, title=sub_head, to=txtf)

            # Summarize this set of fitted params as plots pred-vs-label|control and in text tables
            results_stem = f"fitted-params-from-{prediction_type}-vs-{comp_type}"
            names = [n for n in fit_names if n not in ["ecosw", "esinw"]] + ["ecosw", "esinw"]
            fig = plots.plot_predictions_vs_labels(fitted_vals, comp_vals, trans_flags, names,
                                                   xlabel_prefix=comp_head, ylabel_prefix="fitted")
            fig.savefig(report_dir / f"{results_stem}.eps")
            plt.close()

            with open(report_dir / f"{results_stem}.txt", "w", encoding="utf8") as txtf:
                for (sub_head, mask, rep_names) in sub_reports:
                    if any(mask):
                        predictions_vs_labels_to_table(fitted_vals[mask], comp_vals[mask],
                                                       targs[mask], rep_names, title=sub_head,
                                                       comparison_head=comp_head.capitalize(),
                                                       prediction_head="Fitted", to=txtf)
    return fitted_vals


def base_jktebop_fit_params(task: int,
                            period: float,
                            primary_epoch: float,
                            dat_file_name: str,
                            file_name_stem: str,
                            sector_cfg: Dict[str, any],
                            simulations: int=100) -> Dict[str, any]:
    """
    Get the basic testing set of JKTEBOP task 3, 8 or 9 in file parameters.
    This sets up mainly fixed values for qphot, grav darkening etc.
    but includes setting up suitable simulation iterations

    L3 defaults to zero and fitted

    qphot defaults to zero and not fitted

    However, quad limb darkening algo and coeffs are found by lookup based
    on the stars' masses and radii (for logg) and effective temps, as held in sector_cfg.
    """
    # Calculate star specific LD params
    ld_params = {}
    for star in ["A", "B"]:
        logg = stellar.log_g(sector_cfg[f"M{star}"] * u.solMass, sector_cfg[f"R{star}"] * u.solRad)
        coeffs = limb_darkening.lookup_tess_quad_ld_coeffs(logg, sector_cfg[f"Teff{star}"] * u.K)
        ld_params[f"LD{star}"] = "quad"
        ld_params[f"LD{star}1"] = coeffs[0]
        ld_params[f"LD{star}2"] = coeffs[1]

    if task == 3:
        simulations = "" # The only valid option for task 3

    return {
        "task": task,
        "qphot": 0.,
        "gravA": 0.,        "gravB": 0.,
        "L3": 0.,

        **ld_params,

        "reflA": 0.,        "reflB": 0.,
        "period": period,
        "primary_epoch": primary_epoch,

        "simulations": simulations,

        "qphot_fit": 0,
        "ecosw_fit": 1,     "esinw_fit": 1,
        "L3_fit": 1,
        "LDA1_fit": 1,      "LDB1_fit": 1,
        "LDA2_fit": 0,      "LDB2_fit": 0,
        "period_fit": 1,
        "primary_epoch_fit": 1,
        "data_file_name": dat_file_name,
        "file_name_stem": file_name_stem,
    }


def append_calculated_inc_predictions(preds: np.ndarray[UFloat]) -> np.ndarray[UFloat]:
    """
    Calculate the predicted inc value (in degrees) and append to the predictions
    if not already present.

    :preds: the predictions recarray to which inc should be appended
    :returns: the predictions with inc added if necessary
    """
    names = list(preds.dtype.names)
    if "inc" not in names:
        if "bP" in names:
            # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
            r = preds["rA_plus_rB"] / (1+preds["k"])
            csi = preds["bP"] * r * (1+preds["esinw"]) / (1-(preds["ecosw"]**2 + preds["esinw"]**2))
            inc = np.array([degrees(acos(i)) for i in csi], dtype=UFloat)
        elif "cosi" in names:
            inc = np.array([degrees(acos(val)) for val in preds["cosi"]], dtype=UFloat)
        elif "sini" in names:
            inc = np.array([degrees(asin(val)) for val in preds["sini"]], dtype=UFloat)
        else:
            raise KeyError("Missing bP, cosi or sini in predictions required to calc inc.")

        # It's difficult to append a field to an "object" array or recarray so copy over to new inst
        new = np.empty_like(preds,
                            dtype=np.dtype(preds.dtype.descr + [("inc", np.dtype(UFloat.dtype))]))
        new[names] = preds[names]
        new["inc"] = inc
        return new


def will_transit(rA_plus_rB: Union[np.ndarray[float], np.ndarray[UFloat]],
                 k: Union[np.ndarray[float], np.ndarray[UFloat]],
                 inc: Union[np.ndarray[float], np.ndarray[UFloat]],
                 ecosw: Union[np.ndarray[float], np.ndarray[UFloat]],
                 esinw: Union[np.ndarray[float], np.ndarray[UFloat]]) \
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
    # Stop numpy choking on uncertainties; we only need the nominals to estimate a transit
    rA_plus_rB = unumpy.nominal_values(rA_plus_rB)
    k = unumpy.nominal_values(k)
    inc = unumpy.nominal_values(inc)
    ecosw = unumpy.nominal_values(ecosw)
    esinw = unumpy.nominal_values(esinw)

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


def predictions_vs_labels_to_table(predictions: np.ndarray[UFloat],
                                   labels: np.ndarray[UFloat],
                                   block_headings: np.ndarray[str]=None,
                                   selected_param_names: np.ndarray[str]=None,
                                   reverse_scaling: bool=False,
                                   comparison_head: str="Label",
                                   prediction_head: str="Prediction",
                                   title: str=None,
                                   summary_only: bool=False,
                                   to: TextIOBase=None):
    """
    Will write a text table of the predicted nominal values vs the label values
    with O-C, MAE and MSE metrics, to the requested output.

    :predictions: the predictions
    :labels: the labels
    :block_headings: the heading for each block of preds-vs-labels
    :selected_param_names: a subset of the full list of labels/prediction names to render
    :reverse_scaling: whether to reverse the scaling of the values to represent the model output
    :comparison_head: the text of the comparison row headings (10 chars or less)
    :prediction_head: the text of the prediction row headings (10 chars or less)
    :title: optional title text to write above the table
    :summary_only: omit the body and just report the summary
    :to: the output to write the table to. Defaults to printing.
    """
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    # We output the params common to the all labels & predictions or those requested
    if selected_param_names is None:
        keys = np.array([k for k in labels.dtype.names if k in predictions.dtype.names])
    elif isinstance(selected_param_names, str):
        keys = np.array([selected_param_names])
    else:
        keys = selected_param_names

    resids = calculate_residuals(predictions, labels, keys, reverse_scaling)
    maes = unumpy.nominal_values([np.mean(np.abs(resids[k])) for k in keys])
    mses = unumpy.nominal_values([np.mean(np.square(resids[k])) for k in keys])

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
            block_headings = (f"{n:04d}" for n in range(1, len(predictions)+1))
        elif isinstance(block_headings, str):
            block_headings = [block_headings]

        # Make sure these are iterable; not always the case if we're given a single row
        labels = np.expand_dims(labels, 0) if labels.shape == () else labels
        predictions = np.expand_dims(predictions, 0) if predictions.shape == () else predictions
        resids = np.expand_dims(resids, 0) if resids.shape == () else resids

        for block_head, b_comp, b_preds, b_ocs in zip(block_headings, labels, predictions, resids):
            # A sub table for each block/instance with 3 rows; labels|controls, predictions and ocs
            header_block(block_head)
            horizontal_line("-")
            for row_head, row_vals in zip([comparison_head, prediction_head, "O-C"],
                                          [b_comp, b_preds, b_ocs]):
                to.write(f"{row_head:<10s} | " +
                         " ".join(f"{get_nom(row_vals[k]):10.6f}" for k in keys))
                if row_head == "O-C":
                    to.write(f" {get_nom(np.mean(np.abs(row_vals[keys].tolist()))):10.6f}")
                    to.write(f" {get_nom(np.mean(np.square(row_vals[keys].tolist()))):10.6f}")
                to.write("\n")

    # Summary rows for aggregate stats over all of the rows
    horizontal_line("=")
    to.write(f"{'MAE':<10s} | " + " ".join(f"{get_nom(v):10.6f}" for v in maes) +
             f" {get_nom(np.mean(np.abs(resids.tolist()))):10.6f}\n")
    to.write(f"{'MSE':<10s} | " + " ".join(f"{get_nom(v):10.6f}" for v in mses) +
             " "*11 + f" {get_nom(np.mean(np.square(resids.tolist()))):10.6f}\n")
    if print_it:
        print(to.getvalue())

def get_nom(value):
    """ Get the nominal value if the passed value is a UFloat other return as is """
    return value.nominal_value if isinstance(value, UFloat) else value

def calculate_residuals(predictions: np.ndarray[UFloat],
                        labels: np.ndarray[Union[UFloat, float]],
                        selected_param_names: np.ndarray[str]=None,
                        reverse_scaling: bool=False) -> np.ndarray[UFloat]:
    """
    Calculates the residual by subtracted the predictions from the labels.

    :predictions: the predictions
    :labels: the labels
    :selected_names: subset of the columns, or all predicted columns if None
    :reverse_scaling: whether to reapply label scaling to get the model's values
    :returns: a structured NDArray[UFloat] of the residuals over the selected names
    """
    # We output the params common to the both labels & predictions or those requested
    # (which we allow to error if a requested name is not found)
    if selected_param_names is None:
        dtype = [(n, np.dtype(UFloat.dtype))
                 for n in labels.dtype.names if n in predictions.dtype.names]
    else:
        dtype = [(n, np.dtype(UFloat.dtype)) for n in selected_param_names]

    # Haven't found a way to do the subtract directly on the whole NDarray if they contain UFloats
    # (no subtract in unumpy). We do it a (common) column/param at a time, which has the added
    # benefit of being untroubled by the two arrays having different sets of cols (widths).
    row_count = predictions.shape[0] if predictions.shape else 1
    resids = np.empty(shape=(row_count, ), dtype=dtype)

    # We may have to reverse the scaling
    if reverse_scaling:
        for n in selected_param_names:
            resids[n] = (labels[n] - predictions[n]) * deb_example.labels_and_scales[n]
    else:
        for n in selected_param_names:
            resids[n] = labels[n] - predictions[n]
    return resids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runs formal testing on trained model files.")
    ap.add_argument(dest="model_files", type=Path, nargs="*", help="The model file(s) to test.")
    ap.set_defaults(model_files=[None]) # If None will load the default model under ebop_maven/data
    args = ap.parse_args()

    # This will get the config, labels and published params for formal targets not excluded
    targets_config_file = Path("./config/formal-test-dataset.json")
    formal_targs_cfg = dict(formal_testing.iterate_target_configs(targets_config_file))
    formal_targs = list(formal_targs_cfg.keys())
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
            ctrl_fit_vals = None # To be set on the first, control fit run
            for (pred_type, is_ctrl_fit, iterations) in [
                ("control",     True,       0),
                ("nonmc",       False,      1),
                ("mc",          False,      1000),
            ]:
                print(f"\nTesting JKTEBOP fitting of {pred_type} input values\n" + "="*80)
                fit_vals = fit_against_formal_test_dataset(the_estimator,
                                                           formal_targs_cfg,
                                                           formal_targs,
                                                           iterations,
                                                           True,
                                                           is_ctrl_fit,
                                                           None if is_ctrl_fit else ctrl_fit_vals,
                                                           result_dir)
                if is_ctrl_fit:
                    ctrl_fit_vals = fit_vals
