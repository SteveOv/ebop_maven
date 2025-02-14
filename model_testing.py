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
from datetime import datetime
import warnings

import matplotlib.pylab as plt

from uncertainties import ufloat, UFloat, unumpy
from lightkurve import LightCurve
import astropy.units as u
import numpy as np

import tensorflow as tf
import keras
from keras import Model, layers

from deblib import stellar, limb_darkening, orbital
from deblib.constants import M_sun, R_sun
from deblib.vmath import arccos, arcsin, cos, degrees, radians

from ebop_maven.estimator import Estimator
from ebop_maven import deb_example

from traininglib import datasets, formal_testing, jktebop, pipeline, plots
from traininglib.tee import Tee


FORMAL_TEST_DATASET_DIR = Path("./datasets/formal-test-dataset/")

DEFAULT_TESTING_SEED = 42

# Those params resulting from predictions that we need to pass to JKTEBOP
fit_params = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc"]

# Superset of all of the potentially fitted parameters
all_fitted_params = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc",
                     "L3", "pe", "period", "bP", "bS", "ecc", "omega", "qphot",
                     "phiS", "rA", "rB", "LDA1", "LDB1", "LDA2", "LDB2"]

def evaluate_model_against_dataset(estimator: Union[Path, Model, Estimator],
                                   mc_iterations: int=1,
                                   include_ids: List[str]=None,
                                   test_dataset_dir: Path=FORMAL_TEST_DATASET_DIR,
                                   report_dir: Path=None,
                                   scaled: bool=False):
    """
    Will evaluate the indicated model file or Estimator against the contents of the
    chosen test dataset. The evaluation is carried out in the context of an Estimator,
    rather than with model evaluate, as it gives us more scope to drill into the data.

    :estimator: the Estimator or estimator model to use to make predictions
    :mc_iterations: the number of MC Dropout iterations
    :include_ids: list of target ids to predict, or all if not set
    :test_dataset_dir: the location of the formal test dataset
    :report_dir: optional directory into which to save reports, reports not saved if this is None
    :scaled: whether labels and predictions are scaled (raw model predictions) or not
    """
    # Create our Estimator. It will tell us what its inputs should look like
    # and which labels it (& the underlying model) can predict.
    if not isinstance(estimator, Estimator):
        estimator = Estimator(estimator)
    mc_type = "mc" if mc_iterations > 1 else "nonmc"

    # Need these for any filtering on extra_features; we use the names to find the Tensor indices
    all_feat_names = deb_example.get_all_extra_feature_names()

    # To clarify: the estimator publishes a list of what it can predict via its label_names attrib
    # and fit_params may differ; they are those params required for JKTEBOP fitting and reporting.
    # super_params is the set of both and is used, for example, to get the superset of label values
    # We also need inc as it is used to calculate whether a system is transiting or not
    super_params = estimator.label_names + [n for n in fit_params if n not in estimator.label_names]

    print(f"Looking for the test dataset in '{test_dataset_dir}'...", end="")
    ds_name = test_dataset_dir.name
    tfrecord_files = sorted(test_dataset_dir.glob("**/*.tfrecord"))
    if len(tfrecord_files) == 0:
        raise ValueError(f"No tfrecord files under {test_dataset_dir}. Please make this dataset.")
    print(f"found {len(tfrecord_files)} file(s).")

    # The label values here will have zero uncertainties as we only store the nominals in the ds.
    # We request all of the feature values here as we will want them for filter masks below.
    ids, mags_vals, feat_vals, lbl_vals = datasets.read_dataset(tfrecord_files,
                                                mags_bins=estimator.mags_feature_bins,
                                                mags_wrap_phase=estimator.mags_feature_wrap_phase,
                                                labels=super_params,
                                                identifiers=include_ids,
                                                scale_labels=scaled,
                                                filter_func=None)

    # Mask for subsets of the instances we expect the model to be able to predict well.
    easy_mask = (feat_vals[..., all_feat_names.index("depthP")] > 0.1) \
              & (feat_vals[..., all_feat_names.index("depthS")] > 0.1)

    # Sets the random seed on numpy, keras's backend library (here tensorflow) and python
    keras.utils.set_random_seed(DEFAULT_TESTING_SEED)

    # Make all predictions, returned as a structured array of shape (#insts, #labels) & dtype=UFloat
    # Manually batch large datasets in case the model is memory constrained (i.e.: running on a GPU)
    inst_count = len(ids)
    print(f"The Estimator is making predictions on the {inst_count} test instance(s)",
          f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
    max_batch_size = 1000
    feat_vals = feat_vals[..., [all_feat_names.index(k) for k in estimator.extra_feature_names]]
    pred_vals = np.empty((inst_count, ),
                         dtype=[(n, np.dtype(UFloat.dtype)) for n in estimator.label_names])
    force_seed_on_dropout_layers(estimator)
    for ix in np.arange(0, inst_count, max_batch_size):
        pv = estimator.predict(mags_feature=mags_vals[ix : ix+max_batch_size],
                               extra_features=feat_vals[ix : ix+max_batch_size],
                               iterations=mc_iterations,
                               unscale=not scaled)
        pred_vals[ix : ix+len(pv)] = pv

    if "inc" not in estimator.label_names:
        pred_vals = append_calculated_inc_predictions(pred_vals)
        if scaled: # Apply any scaling to inc consistent with having predicted it
            pred_vals["inc"] *= deb_example.labels_and_scales.get("inc", 1)

    if scaled:
        print("The predictions & labels are scaled with the following values:\n\t" +
            ", ".join(f"{k} = {deb_example.labels_and_scales.get(k, 1):.3f}" for k in super_params))

    # Work out which are the transiting systems so we can break down the reporting
    pnames = list(inspect.signature(will_transit).parameters)
    argvs = [lbl_vals[p] / (deb_example.labels_and_scales[p] if scaled else 1) for p in pnames]
    tran_mask = will_transit(*argvs)

    # Now report. If the labels are read from the dataset/tfrecord they will have no uncertainties.
    # Skip some tables/plots for the formal-test-ds as it is too small for them to be meaningful.
    plot_params = [n for n in estimator.label_names if n not in ["ecosw","esinw"]]+["ecosw","esinw"]
    show_error_bars = mc_iterations > 1
    for (subset,            s_mask,             do_synth_rprt,  do_frml_tbl,    do_frml_rprt) in [
        ("",                [True]*inst_count,  True,           True,           True),  # all
        (" transiting",     tran_mask,          True,           True,           False),
        (" non-transiting", ~tran_mask,         True,           True,           False),
        (" easier",         easy_mask,          False,          False,          False),
        (" harder",         ~easy_mask,         False,          False,          False),
    ]:
        if any(s_mask):
            # Slightly fiddly; each iteration's preds/labels subset is picked out with s_mask.
            # We may further subdivide the subset using the various trans/easy/hard masks, and
            # these masks will requiring masking with s_mask before they're used on the subset.
            s_preds, s_lbls, s_tran_mask = pred_vals[s_mask], lbl_vals[s_mask], tran_mask[s_mask]
            suffix = subset.replace(' ','-')
            if (("synth" in ds_name) or ("formal" in ds_name and do_frml_tbl)):
                print(f"\nSummary of the estimator predictions for {sum(s_mask)}{subset} system(s)")
                predictions_vs_labels_to_table(s_preds, s_lbls, summary_only=True,
                                               selected_param_names=estimator.label_names,
                                               error_bars=show_error_bars)

                if set(fit_params) != set(estimator.label_names):
                    print("...and of the corresponding fitting input params derived from them")
                    predictions_vs_labels_to_table(s_preds, s_lbls, summary_only=True,
                                                   selected_param_names=fit_params,
                                                   error_bars=show_error_bars)

            # Use pdf as usually smaller file size than eps and also supports transparency/alpha.
            if report_dir and (("synth" in ds_name and do_synth_rprt) \
                                or ("formal" in ds_name and do_frml_rprt)):
                s_dir = report_dir / ds_name / mc_type
                s_dir.mkdir(parents=True, exist_ok=True)
                save_predictions_to_csv(ids, s_tran_mask, s_preds, s_dir/f"predictions{suffix}.csv")
                save_predictions_to_csv(ids, s_tran_mask, s_lbls, s_dir/f"labels{suffix}.csv")

                # Box plot of the error distributions for each predicted params
                fliers = "formal" in ds_name
                s_errs = calculate_prediction_errors(s_preds[plot_params], s_lbls[plot_params])
                err_sets = [s_errs[easy_mask[s_mask]], s_errs[~easy_mask[s_mask]]]
                fig = plots.plot_prediction_boxplot(err_sets, show_fliers=fliers, ylabel="Error")
                fig.savefig(s_dir / f"predictions-{mc_type}-box{suffix}.pdf")
                fig.clf()

                # If we have a very large dataset then adopt a strategy of skipping data in the plot
                # as it's worthless plotting every one as individual datapoints become meaningless
                # with the plot area being small. This will help keep the file size under control.
                sl = slice(0, None, int(np.ceil(inst_count / 10000)))
                fig = plots.plot_predictions_vs_labels(s_preds[sl], s_lbls[sl], s_tran_mask[sl],
                                                       plot_params, show_errorbars=show_error_bars,
                                                       hl_mask2=easy_mask[s_mask][sl])
                fig.savefig(s_dir / f"predictions-{mc_type}-vs-labels{suffix}.pdf")
                fig.clf()


def fit_formal_test_dataset(estimator: Union[Path, Model, Estimator],
                            targets_config: Dict[str, any],
                            include_ids: List[str]=None,
                            mc_iterations: int=1,
                            apply_fit_overrides: bool=True,
                            do_control_fit: bool=False,
                            comparison_vals: np.ndarray[UFloat]=None,
                            report_dir: Path=None) -> np.ndarray[UFloat]:
    """
    Will fit members of the formal test dataset, as configured in targets_config, based on
    predictions made with the passed estimator model, returning the corresponding fitted params
    (except when do_control_fit==True when the label values are used instead of predictions).

    Unlike evaluate_model_against_dataset() these tests do not use the dataset tfrecord files,
    rather the targets' mags_features are sourced and pre-processed directly from the TESS fits
    files so that these tests and the equivalent logic in model_interactive_tester are similar.
    This means that MC Dropout predictions may vary from those of evaluate_model_against_dataset()
    because that method performs a single bulk predict on all chosen targets in the dataset
    (the mags_features should be the same, so it's down to the random behaviour of MC Dropout).

    :estimator: the Estimator or estimator model to use to make predictions
    :targets_config: the full config for all targets
    :include_ids: list of target ids to fit, or all in targets_config if not given
    :mc_iterations: the number of MC iterations to use when making predictions
    :apply_fit_overrides: apply any fit_overrides from each target's config
    :do_control_fit: when True labels, rather than predictions, will be the input params for fitting
    :comparison_vals: optional recarray[UFloat] to compare to fitting results, alternative to labels
    :report_dir: optional directory into which to save reports, reports not saved if this is None
    :returns: a structured NDArray[UFloat] containing the fitted parameters for each target
    """
    # pylint: disable=too-many-statements, too-many-branches
    if not isinstance(estimator, Estimator):
        estimator = Estimator(estimator)
    mags_bins = estimator.mags_feature_bins
    mags_wrap_phase = estimator.mags_feature_wrap_phase

    targs = include_ids if len(include_ids or []) > 0 else [*targets_config.keys()]
    targs = np.array(targs) if not isinstance(targs, np.ndarray) else targs
    trans_flags = np.array([targets_config.get(t, {}).get("transits", False) for t in targs])
    prediction_type = "control" if do_control_fit else "mc" if mc_iterations > 1 else "nonmc"

    # Highlighting with plot_predictions_vs_labels(): AI Phe (tough fit), V570 Per (tough to pred)
    hl_mask1 = targs == "AI Phe"
    hl_mask2 = targs == "V570 Per"

    # To clarify: the estimator publishes a list of what it can predict via its label_names attrib
    # and fit_params may differ; they are those params required for JKTEBOP fitting and reporting.
    # super_params is the set of both and is used, for example, to get the superset of label values
    super_params = estimator.label_names + [n for n in fit_params if n not in estimator.label_names]

    # For this "deep dive" test we report on labels with uncertainties, so we ignore the label
    # values in the dataset (nominals only) and go to the source config to get the full values.
    lbl_vals = formal_testing.get_labels_for_targets(targets_config, super_params, targs)

    # Pre-allocate a structured arrays to hold the predictions and equivalent fit results
    pred_vals = np.empty((len(targs), ), dtype=[(p, np.dtype(UFloat.dtype)) for p in super_params])
    fit_vals = np.empty((len(targs), ), dtype=[(p, np.dtype(UFloat.dtype)) for p in super_params])

    # Pre-allocate the mags feature and equivalent LC from predicted and fitted parameters
    mags_feats = np.empty((len(targs), mags_bins), dtype=float)
    pred_feats = np.empty((len(targs), 1001), dtype=float)
    fit_feats = np.empty((len(targs), 1001), dtype=float)

    # Finally, we have everything in place to fit our targets and report on the results
    for ix, targ in enumerate(targs):
        print(f"\n\nProcessing target {ix + 1} of {len(targs)}: {targ}\n" + "-"*40)
        targ_config = targets_config[targ].copy()
        print(fill(targ_config.get("desc", "")) + "\n")

        # The basic lightcurve data read, rectified & extended with delta_mag and delta_mag_err cols
        (lc, _) = formal_testing.prepare_lightcurve_for_target(targ, targ_config, True)
        period = targ_config["period"] * u.d
        pe = pipeline.to_lc_time(targ_config["primary_epoch"], lc)

        # Work out how we will position/wrap the phase folded mags feature
        wrap_phase = mags_wrap_phase
        if wrap_phase is None:
            ecosw = lbl_vals[ix]["ecosw"].nominal_value
            wrap_phase = 0.5+(orbital.phase_of_secondary_eclipse(ecosw, targ_config.get("ecc",0))/2)

        # Get the phase folded and binned mags feature
        print(f"Creating a folded & phase-normalized light curve about {pe.format} {pe} & {period}",
              f"wrapped beyond phase {wrap_phase}" if wrap_phase not in [0.0, 1.0] else "")
        fold_lc = lc.fold(period, pe, wrap_phase=u.Quantity(wrap_phase), normalize_phase=True)
        _, mags = pipeline.get_sampled_phase_mags_data(fold_lc, mags_bins, wrap_phase)

        if do_control_fit:
            pred_vals[ix] = copy.deepcopy(lbl_vals[ix])
        else:
            print(f"\nThe Estimator will make {prediction_type} predictions on {targ}",
                  f"with {mc_iterations} MC Dropout iterations" if prediction_type == "mc" else "")
            keras.utils.set_random_seed(DEFAULT_TESTING_SEED)
            force_seed_on_dropout_layers(estimator)
            pv = estimator.predict(np.array([mags]), None, mc_iterations)
            predictions_vs_labels_to_table(pv, lbl_vals[ix], [targ])
            pred_vals[ix] = pv if "inc" in pv.dtype.names else append_calculated_inc_predictions(pv)

        print(f"\nThe {prediction_type} sourced input params for fitting {targ}")
        predictions_vs_labels_to_table(pred_vals[ix], lbl_vals[ix], [targ], fit_params)

        # Perform the task3 fit taking the preds or control as input params and supplementing
        # them with parameter values and fitting instructions from the target's config.
        fit_stem = "model-testing-" + re.sub(r"[^\w\d-]", "-", targ.lower())
        fit_vals[ix] = fit_target(lc, targ, pred_vals[ix], targ_config, super_params, fit_stem,
                                  task=3, apply_fit_overrides=apply_fit_overrides)

        print(f"\nHave fitted {targ} resulting in the following fitted params")
        predictions_vs_labels_to_table(fit_vals[ix], lbl_vals[ix], [targ], fit_params,
                                       prediction_head="Fitted")

        # Get the phase-folded mags data for the mags feature, predicted and actual fit.
        # We need to undo the wrap of the mags_feature as the plot will apply its own fixed wrap.
        mags_feats[ix] = np.roll(mags, -int((1-wrap_phase) * len(mags)), axis=0)
        pred_lc = generate_predicted_fit(pred_vals[ix], targ_config, apply_fit_overrides, 1001)
        pred_feats[ix] = pred_lc["delta_mag"]
        with open(jktebop.get_jktebop_dir() /f"{fit_stem}.fit", mode="r", encoding="utf8") as ff:
            fit = np.loadtxt(ff, usecols=[1], comments="#", dtype=float)
            fit_feats[ix] = fit[np.round(np.linspace(0, fit.shape[0]-1, 1001)).astype(int)]

    # Save reports on how the predictions and fitting has gone over all of the selected targets
    # Publication plots use pdf as this usually gives smaller file sizes than eps & supports alpha.
    if report_dir:
        sub_dir = report_dir / prediction_type
        sub_dir.mkdir(parents=True, exist_ok=True)
        save_predictions_to_csv(targs, trans_flags, pred_vals, sub_dir / "predictions.csv")
        save_predictions_to_csv(targs, trans_flags, fit_vals, sub_dir / "fitted-params.csv")
        save_predictions_to_csv(targs, trans_flags, lbl_vals, sub_dir / "labels.csv")

        comparison_type = [("labels", "label", lbl_vals)] # type, heading, values
        if comparison_vals is not None:
            comparison_type += [("control", "control", comparison_vals)]
            save_predictions_to_csv(targs, trans_flags, comparison_vals, sub_dir / "controls.csv")
        sub_reports = [
            ("All targets (model labels)",          [True]*len(targs),      estimator.label_names),
            ("\nTransiting systems only",           trans_flags,            estimator.label_names),
            ("\nNon-transiting systems only",       ~trans_flags,           estimator.label_names),
            ("\n\n\nAll targets (fitting params)",  [True]*len(targs),      fit_params),
            ("\nTransiting systems only",           trans_flags,            fit_params),
            ("\nNon-transiting systems only",       ~trans_flags,           fit_params)]

        for comp_type, comp_head, comp_vals in comparison_type:
            # Summarize this set of predictions as plots-vs-label|control and in text table
            # Control == fit from labels not preds, so no point producing these
            if not do_control_fit:
                preds_stem = f"predictions-{prediction_type}-vs-{comp_type}"
                for (source, pnames) in [("model", estimator.label_names), ("fitting", fit_params)]:
                    pnames = [n for n in pnames if n not in ["ecosw", "esinw"]] + ["ecosw", "esinw"]
                    fig = plots.plot_predictions_vs_labels(pred_vals, comp_vals, trans_flags,
                                                           pnames, xlabel_prefix=comp_head,
                                                           hl_mask1=hl_mask1, hl_mask2=hl_mask2)
                    fig.savefig(sub_dir / f"{preds_stem}-{source}.pdf")

                with open(sub_dir / f"{preds_stem}.txt", mode="w", encoding="utf8") as txtf:
                    for (sub_head, mask, rep_names) in sub_reports:
                        if any(mask):
                            predictions_vs_labels_to_table(pred_vals[mask], comp_vals[mask],
                                                    targs[mask], rep_names, title=sub_head,
                                                    error_bars=prediction_type == "mc", to=txtf)

            # Summarize this set of fitted params as plots pred-vs-label|control and in text tables
            results_stem = f"fitted-params-from-{prediction_type}-vs-{comp_type}"
            pnames = [n for n in fit_params if n not in ["ecosw", "esinw"]] + ["ecosw","esinw"]
            fig = plots.plot_predictions_vs_labels(fit_vals, comp_vals, trans_flags, pnames,
                                                   xlabel_prefix=comp_head, ylabel_prefix="fitted",
                                                   hl_mask1=hl_mask1, hl_mask2=hl_mask2)
            fig.savefig(sub_dir / f"{results_stem}.pdf")
            plt.close()

            if not do_control_fit:
                # Plot out the input feature vs predicted fit vs actual fit for each test system.
                # This can get very large, so we can split it into multiple plots with slices.
                for ix, sl in enumerate([slice(0, 25)], start=1):
                    fig = plots.plot_folded_lightcurves(mags_feats[sl], targs[sl], pred_feats[sl],
                                                        fit_feats[sl], extra_names=(None, None),
                                                        init_ymax=1., extra_yshift=0.2, cols=5)
                    fig.savefig(sub_dir / f"fold-mags-from-{prediction_type}-pt-{ix}.pdf")
                    plt.close()

            with open(sub_dir / f"{results_stem}.txt", "w", encoding="utf8") as txtf:
                for (sub_head, mask, rep_names) in sub_reports:
                    if any(mask):
                        predictions_vs_labels_to_table(fit_vals[mask], comp_vals[mask], targs[mask],
                                        rep_names, title=sub_head, prediction_head="Fitted",
                                        label_head=comp_head.capitalize(), error_bars=True, to=txtf)
    return fit_vals


def fit_target(lc: LightCurve,
               target: str,
               input_params: np.ndarray[UFloat],
               target_cfg: dict[str, any],
               return_keys: List[str] = None,
               file_stem: str = "model-testing-",
               task: int=3,
               simulations: int=100,
               apply_fit_overrides: bool=True,
               retries: int=1) -> np.ndarray[UFloat]:
    """
    Perform a JKTEBOP fitting on the passed light-curve based on the input_params and target config
    passed in. This covers the following tasks;
    - lookup quad limb darkening coeffs based on the M(A|B), R(A|B) and Teff(A|B) config values
    - write the JKTEBOP in file with the
      - task, default parameter values, period & p. e. from config and above limb darkening params
      - overridden with the input_params
      - optionally overriden with the fit_overrides from the config, including lrat instructions
      - sf (scale factor) 1st order poly instructions for each contiguous stretch of LC on 1 day gap
      - chif instruction to adjust error bars to give chi^2 of 1.0, after fitting
    - write the JKTEBOP dat file from the light-curve's time, delta_mag and delta_mag_err fields
    - invoke JKTEBOP to process the in and dat file
      - retry the fit, from where the current fit stopped, if a warning indicating a "good fit not
      found after ### iterations" is raised and available retries > 0
      - if we've run out of retries and we're still getting the warning, select the fitted values
      from the first attempt for parsing as the result
    - parse the resulting par file to read & return the requested return_keys' values

    :lc: the light-curve data to fit
    :target: the name of the target system
    :input_params: the initial values for the fitted params to override the default values
    :target_cfg: the target config dictionary containing labels, characteristics and overrides
    :return_keys: keys for the fitted parameters to populate the return array, or all if None
    :file_stem: the file name stem for each JKTEBOP file written
    :task: the JKTEBOP task to execute; 3, 8 or 9
    :simulations: the number of simulations to run when task 8 or 9 (ignored for task 3)
    :apply_fit_overrides: whether to apply any fit_overrides from the target config
    :retries: number of times to retry on failure to converge on a good fit
    :returns: a structured NDArray[UFloat] of those return_keys found in the fitted par file
    """
    if return_keys is None:
        return_keys = all_fitted_params

    sector_count = sum(1 for s in target_cfg["sectors"] if s.isdigit())
    print(f"\nFitting {target} (with {sector_count} sector(s) of data) with JKTEBOP task {task}...")

    fit_dir = jktebop.get_jktebop_dir()
    in_fname = fit_dir / f"{file_stem}.in"
    dat_fname = fit_dir / f"{file_stem}.dat"
    par_fname = fit_dir / f"{file_stem}.par"

    # The fit_overrides are optional overrides to any derived value and should be applied last
    fit_overrides = copy.deepcopy(target_cfg.get("fit_overrides",{})) if apply_fit_overrides else {}
    ld_params = pop_and_complete_ld_config(fit_overrides, target_cfg) # leaves LD*_fit items

    attempts = 1 + max(0, retries)
    best_attempt = 0
    fitted_params = np.empty(shape=(attempts, ),
                             dtype=[(k, np.dtype(UFloat.dtype)) for k in all_fitted_params])
    for attempt in range(attempts):
        if input_params.shape == (1,):
            input_params = input_params[0]

        all_in_params = {
            "task": task,
            "qphot": 0.,
            "gravA": 0.,                "gravB": 0.,
            "L3": 0.,

            "reflA": 0.,                "reflB": 0.,
            "period": target_cfg["period"],
            "primary_epoch": pipeline.to_lc_time(target_cfg["primary_epoch"], lc).value,

            "simulations": simulations if task == 8 else "",

            "qphot_fit": 0,
            "ecosw_fit": 1,             "esinw_fit": 1,
            "gravA_fit": 0,             "gravB_fit": 0,
            "L3_fit": 1,
            "LDA1_fit": 1,              "LDB1_fit": 1,
            "LDA2_fit": 0,              "LDB2_fit": 0,
            "reflA_fit": 1,             "reflB_fit": 1,
            "sf_fit": 1,
            "period_fit": 1,            "primary_epoch_fit": 1,

            "data_file_name": dat_fname.name,
            "file_name_stem": file_stem,

            **{ n: input_params[n] for n in input_params.dtype.names },
            **fit_overrides,
            **ld_params,
        }

        # Add scale-factor poly fitting, chi^2 adjustment (to 1.0) and any light-ratio instructions
        # The lrats are spectroscopic light ratios which may be specified to constrain fitting and
        # the chif instruction tells JKTEBOP to adjust the fit's error bars until the chi^2 == 1.
        segments = pipeline.find_lightcurve_segments(lc, 0.5, return_times=True)
        append_lines = jktebop.build_poly_instructions(segments, "sf", 1)
        lrats = fit_overrides.get("lrat", []) if fit_overrides else []
        append_lines += ["", "chif", ""] + [ f"lrat {l}" for l in lrats ]

        # JKTEBOP will fail if it finds files from a previous fitting
        for file in fit_dir.glob(f"{file_stem}.*"):
            file.unlink()
        jktebop.write_in_file(in_fname, append_lines=append_lines, **all_in_params)
        jktebop.write_light_curve_to_dat_file(lc, dat_fname)

        # Warnings are a mess! I haven't found a way to capture a specific type of warning with
        # specified text and leave everything else to behave normally. This is the nearest I can
        # get but it seems to suppress reporting all warnings, which is "sort of" OK as it's a
        # small block of code and I don't expect anythine except JktebopWarnings to be raised here.
        with warnings.catch_warnings(record=True, category=jktebop.JktebopTaskWarning) as warn_list:
            # Context manager will list any JktebopTaskWarning raised in this context

            # Blocks on the JKTEBOP task until we can parse the newly written par file contents
            # to read out the revised values for the superset of potentially fitted parameters.
            pgen = jktebop.run_jktebop_task(in_fname, par_fname, stdout_to=sys.stdout)
            for k, v in jktebop.read_fitted_params_from_par_lines(pgen, all_fitted_params).items():
                fitted_params[attempt][k] = ufloat(v[0], v[1])

            if attempts > 1 \
                    and sum(1 for w in warn_list if "good fit was not found" in str(w.message)):
                if attempt+1 < attempts: # Further attempts available
                    print(f"Attempt {attempt+1} didn't fully converge on a good fit. {attempts}",
                        "attempt(s) allowed so will retry from the final position of this attempt.")
                    input_params = fitted_params[attempt].copy()
                else:
                    print(f"Failed to fully converge on a good fit after {attempt+1} attempts.",
                          f"Reverting to the results from attempt {best_attempt+1}.")
                    break
            else: # Retries are off or the fit worked
                best_attempt = attempt
                break

    return fitted_params[best_attempt][return_keys]


def generate_predicted_fit(input_params: np.ndarray[UFloat],
                           target_cfg: dict[str, any],
                           apply_fit_overrides: bool=True,
                           bins: int=None) -> np.ndarray[float]:
    """
    Will generate a phase-folded model light curve for the passed params and any
    LD algo/coefficients in the target config.

    :input_params: the param set to use to generate the model LC
    :target_cfg: the full config for this target - allows access to fit_overrides
    :apply_fit_overrides: whether we should use or ignore the contents of fit_overrides
    :bins: optionally indicate the number of bins to return
    :returns: the model data as a numpy structured array of shape (#rows, ["phase", "delta_mag])
    """
    # The fit_overrides are optional overrides to any derived value and should be applied last
    fit_overrides = copy.deepcopy(target_cfg.get("fit_overrides",{})) if apply_fit_overrides else {}
    ld_params = pop_and_complete_ld_config(fit_overrides, target_cfg)

    # We only need a small subset of the params here as we're not fitting, but generating a model LC
    params = {
        "L3":       0,

        # Default to 1, as setting to 0 or -1 leads to jktebop (43) generated light curves
        # showing pronounced dips between eclipses for very close systems (k >~ 0.4).
        "qphot":    1.0,

        **{ n: input_params[n] for n in input_params.dtype.names },
        **fit_overrides,
        **ld_params,
    }

    fit = jktebop.generate_model_light_curve("model-testing-pred-fit", **params)
    length = fit.shape[0]
    if bins is None or bins == length:
        return fit
    return fit[np.round(np.linspace(0, length-1, bins)).astype(int)]


def pop_and_complete_ld_config(source_cfg: Dict[str, any],
                               target_cfg: Dict[str, any]) -> Dict[str, any]:
    """
    Will set up the limb darkening algo and coeffs, first by popping them from
    the source_cfg dictionary then completing the config with missing values.
    Where missing, the algo defaults to quad unless pow2, h1h2 or same specified
    and coefficient lookups are performed to populate any missing values.

    NOTE: pops the LD* items from source_cfg (except those ending _fit) into the returned config

    :source_cfg: the config fragment which may contain predefined LD params
    :target_cfg: the full target config dictionary, which may contain params needed for lookups
    :return: the LD params only dict
    """
    ld_params = {}
    for ld in [k for k in source_cfg if k.startswith("LD") and not k.endswith("_fit")]:
        ld_params[ld] = source_cfg.pop(ld)

    for star in ["A", "B"]:
        algo = ld_params.get(f"LD{star}", "quad") # Only quad, pow2 or h1h2 supported
        if f"LD{star}" not in ld_params \
                or f"LD{star}1" not in ld_params or f"LD{star}2" not in ld_params:
            # If we've not been given overrides for both the algo and coeffs we can look them up
            # provided we have the stellar mass (M?), radius (R?) & effective temp (Teff?) in config
            logg = stellar.log_g(target_cfg[f"M{star}"] * M_sun, target_cfg[f"R{star}"] * R_sun).n
            teff = target_cfg[f"Teff{star}"]
            if algo.lower() == "same":
                coeffs = (0, 0) # JKTEBOP uses the A star params for both
            elif algo.lower() == "quad":
                coeffs = limb_darkening.lookup_quad_coefficients(logg, teff)
            else:
                coeffs = limb_darkening.lookup_pow2_coefficients(logg, teff)

            # Add any missing algo/coeffs tags to the overrides
            ld_params.setdefault(f"LD{star}", algo)
            if algo.lower() == "h1h2":
                # The h1h2 reparameterisation of the pow2 law addreeses correlation between the
                # coeffs; see Maxted (2018A&A...616A..39M) and Southworth (2023Obs...143...71S)
                ld_params.setdefault(f"LD{star}1", 1 - coeffs[0]*(1 - 2**(-coeffs[1])))
                ld_params.setdefault(f"LD{star}2", coeffs[0] * 2**(-coeffs[1]))
            else:
                ld_params.setdefault(f"LD{star}1", coeffs[0])
                ld_params.setdefault(f"LD{star}2", coeffs[1])
    return ld_params


def append_calculated_inc_predictions(preds: np.ndarray[UFloat]) -> np.ndarray[UFloat]:
    """
    Calculate the predicted inc value (in degrees) and append to the predictions
    if not already present.

    :preds: the predictions recarray to which inc should be appended
    :returns: the predictions with inc added if necessary
    """
    names = list(preds.dtype.names)
    if "bP" in names:
        # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
        r1 = preds["rA_plus_rB"] / (1+preds["k"])
        e_squared = preds["ecosw"]**2 + preds["esinw"]**2
        cosi = np.clip(preds["bP"] * r1 * (1+preds["esinw"]) / (1-e_squared),
                       ufloat(-1, 0), ufloat(1, 0))
        inc = degrees(arccos(cosi))
    elif "cosi" in names:
        cosi = np.clip(preds["cosi"], ufloat(-1, 0), ufloat(1, 0))
        inc = degrees(arccos(cosi))
    elif "sini" in names:
        sini = np.clip(preds["sini"], ufloat(-1, 0), ufloat(1, 0))
        inc = degrees(arcsin(sini))
    else:
        raise KeyError("Missing bP, cosi or sini in predictions required to calc inc.")

    if "inc" not in names:
        # It's difficult to append a field to an "object" array or recarray so copy over to new inst
        new = np.empty_like(preds,
                            dtype=np.dtype(preds.dtype.descr + [("inc", np.dtype(UFloat.dtype))]))
        new[names] = preds[names]
    else:
        new = preds.copy()
    new["inc"] = inc
    return new


def primary_impact_param(rA_plus_rB: Union[np.ndarray[float], np.ndarray[UFloat], float, UFloat],
                         k: Union[np.ndarray[float], np.ndarray[UFloat], float, UFloat],
                         inc: Union[np.ndarray[float], np.ndarray[UFloat], float, UFloat],
                         ecosw: Union[np.ndarray[float], np.ndarray[UFloat], float, UFloat],
                         esinw: Union[np.ndarray[float], np.ndarray[UFloat], float, UFloat]) \
                            -> Union[np.ndarray[float], np.ndarray[UFloat], float, UFloat]:
    """
    Calculate the primary impact parameter for the passed label values.

    :rA_plus_rB: the systems' sum of the radii
    :k: the systems' ratio of the radii
    :inc: the orbital inclinations in degrees
    :ecosw: the e*cos(omega) Poincare elements
    :esinw: the e*sin(omega) Poincare elements
    :returns: the calculated primary impact parameter(s)
    """
    # bp = (1/rA) * cos(inc) * (1-e^2 / 1+esinw) where rA = rA_plus_rB/(1+k)
    one_over_rA = (1 + k) / rA_plus_rB
    e = (ecosw**2 + esinw**2)**0.5
    if isinstance(inc, np.ndarray):
        cosi = np.array([cos(radians(i)) for i in inc])
    else:
        cosi = cos(radians(inc))
    return one_over_rA * cosi * (1 - e**2) / (1 + esinw)


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


def save_predictions_to_csv(target_names: np.ndarray[str],
                            flags: np.ndarray[Union[int, bool]],
                            predictions: np.ndarray[UFloat],
                            file_name: Path):
    """
    Writes a csv file for a predictions or labels array. It will include an initial column of
    the target names, a column of the flags, followed by columns for the values split into the
    nominal value (named '{param_name}') and the std_dev error value (named '{param_name}_err').
    The zeroth row is a header row with the column names.

    :target_names: names of the targets, one per row
    :flags: any flags associated with each row; must be a single column of bools or ints (bitmask)
    :predictions: the predictions
    :file_name: the file to (over)write to
    """
    # Life is too short to spend time fighting numpy when trying to split the ufloats, concatenate
    # all of the the columns (with different dtypes) and then use savetxt(), however I did try :-(
    with open(file_name, mode="w", encoding="utf8") as f:
        pkeys = list(predictions.dtype.names)
        f.write("target,flags,")
        f.write(",".join(f"{k:s},{k:s}_err" for k in pkeys))
        f.write("\n")

        # Will error if the zeroth dims are not of equal length
        for targ, flag, preds in zip(target_names, flags, predictions):
            f.write(f"'{targ:s}',{flag:d},")
            f.write(",".join(f"{get_nom(preds[k]):.9f},{get_err(preds[k]):.9f}" for k in pkeys))
            f.write("\n")


def predictions_vs_labels_to_table(predictions: np.ndarray[UFloat],
                                   labels: np.ndarray[UFloat],
                                   block_headings: np.ndarray[str]=None,
                                   selected_param_names: np.ndarray[str]=None,
                                   prediction_head: str="Prediction",
                                   label_head: str="Label",
                                   title: str=None,
                                   summary_only: bool=False,
                                   error_bars: bool=False,
                                   format_dp: int=6,
                                   to: TextIOBase=None):
    """
    Will write a text table of the predicted nominal values vs the label values
    with individual error, MAE and MSE metrics, to the requested output.

    :predictions: the predictions
    :labels: the labels or other comparison values
    :block_headings: the heading for each block of preds-vs-labels
    :selected_param_names: a subset of the full list of labels/prediction names to render
    :prediction_head: the text of the prediction row headings (10 chars or less)
    :label_head: the text of the label/comparison row headings (10 chars or less)
    :title: optional title text to write above the table
    :summary_only: omit the body and just report the summary
    :error_bars: include error bars in output
    :format_dp: the number of decimal places in numeric output. Set <= 6 to maintain column widths
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

    errors = calculate_prediction_errors(predictions, labels, keys)

    # Make sure these are iterable; not always the case if we're given a single row
    labels = np.expand_dims(labels, 0) if labels.shape == () else labels
    predictions = np.expand_dims(predictions, 0) if predictions.shape == () else predictions
    errors = np.expand_dims(errors, 0) if errors.shape == () else errors

    print_it = not to
    if print_it:
        to = StringIO()

    line_length = 13 + (11 * len(keys))-1 + 22
    def horizontal_line(char):
        to.write(char*line_length + "\n")

    def header_block(header):
        horizontal_line("-")
        col_heads = np.concatenate([keys, ["MAE", "MSE"]])
        to.write(f"{header:<10s} | " + " ".join(f"{h:>10s}" for h in col_heads) + "\n")

    num_fmt = f"{{:10.{format_dp:d}f}}"
    def row(row_head, values):
        to.write(f"{row_head:<10s} | ")
        to.write(" ".join(" "*10 if v is None else num_fmt.format(get_nom(v)) for v in values))
        to.write("\n")
        if error_bars:
            to.write(f"{' +/- ':<10s} | ")
            to.write(" ".join(" "*10 if v is None else num_fmt.format(get_err(v)) for v in values))
            to.write("\n")

    if title:
        to.write(f"{title}\n")

    if summary_only:
        header_block("Summary")
    else:
        if block_headings is None or len(block_headings) == 0:
            block_headings = (f"{n:04d}" for n in range(1, len(predictions)+1))
        elif isinstance(block_headings, str):
            block_headings = [block_headings]

        # A sub table for each block/instance with heads & 3 rows; labels|controls, preds and errs
        for block_head, b_lbls, b_preds, b_errs \
                in zip(block_headings, labels, predictions, errors, strict=True):
            header_block(block_head)
            horizontal_line("-")
            for row_head, row_vals in zip([label_head, prediction_head, "Error"],
                                          [b_lbls, b_preds, b_errs]):
                vals = row_vals[keys].tolist()
                if row_head == "Error":
                    vals = np.concatenate([vals, [np.mean(np.abs(vals)), np.mean(np.square(vals))]])
                row(row_head, vals)

    if summary_only or len(predictions) > 1:
        # Summary rows for aggregate stats over all of the rows
        horizontal_line("=")
        key_maes = [np.mean(np.abs(errors[k])) for k in keys]
        key_mses = [np.mean(np.square(errors[k])) for k in keys]
        overall_mae = np.mean(np.abs(errors[keys].tolist()))
        overall_mse = np.mean(np.square(errors[keys].tolist()))
        row("MAE", np.concatenate([key_maes, [overall_mae]]))
        row("MSE", np.concatenate([key_mses, [None, overall_mse]]))

    else:
        horizontal_line("-")

    if print_it:
        print(to.getvalue())

def get_nom(value):
    """ Get the nominal value if the passed value is a UFloat other return value as is """
    return value.nominal_value if isinstance(value, UFloat) else value

def get_err(value):
    """ Get the errorbar (stddev) value if the passed value is a UFloat otherwise err is zero """
    return value.std_dev if isinstance(value, UFloat) else 0.

def calculate_prediction_errors(predictions: np.ndarray[UFloat],
                                labels: np.ndarray[Union[UFloat, float]],
                                selected_param_names: np.ndarray[str]=None) -> np.ndarray[UFloat]:
    """
    Calculates the prediction errors by subtracting the predictions from the label values.

    :predictions: the prediction values
    :labels: the label values
    :selected_param_names: subset of the columns, or the intersection of prediction & label columns
    :returns: a structured NDArray[UFloat] of the residuals over the selected names
    """
    # We output the params common to the both labels & predictions or those requested
    # (which we allow to error if a requested name is not found)
    if selected_param_names is None:
        selected_param_names = [n for n in labels.dtype.names if n in predictions.dtype.names]
    elif isinstance(selected_param_names, str):
        selected_param_names = np.array([selected_param_names])

    # Haven't found a way to do the subtract directly on the whole NDarray if they contain UFloats
    # (no subtract in unumpy). We do it a (common) column/param at a time, which has the added
    # benefit of being untroubled by the two arrays having different sets of cols (widths).
    errors = np.empty(shape=(predictions.shape[0] if predictions.shape else 1, ),
                      dtype=[(n, np.dtype(UFloat.dtype)) for n in selected_param_names])
    for n in selected_param_names:
        errors[n] = labels[n] - predictions[n]
    return errors


def force_seed_on_dropout_layers(estimator: Estimator, seed: int=DEFAULT_TESTING_SEED):
    """
    Forces a seed onto the dropout layers of the model wrapped by the passed Estimator.
    Setting this is a way of making subsequent MC Dropout predictions repeatable.
    Definitely not for "live" but may be useful for testing where repeatability is required.
    
    :estimator: the estimator to modify
    :seed: the new seed value to assign
    """
    # pylint: disable=protected-access
    dropout_layers = (l for l in estimator._model.layers if isinstance(l, layers.Dropout))
    for ix, layer in enumerate(dropout_layers, start=1):
        sg = layer.seed_generator
        new_seed = sg.backend.convert_to_tensor(np.array([0, seed*ix], dtype=sg.state.dtype))
        sg.state.assign(new_seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runs formal testing on trained model files.")
    ap.add_argument("-nf", "--no-fit", dest="do_fit", action="store_false", required=False,
                    help="optionally suppress the fit tests")
    ap.add_argument("-ss", "--synth-suffix", dest="synth_suffix", type=str, required=False,
                    help="optional suffix of /synthetic-mist-test-dataset{SYNTH_SUFFIX}/ directory")
    ap.add_argument(dest="model_files", type=Path, nargs="*", help="The model file(s) to test.")
    # We use a None in model_files as indication to pull in the default model under ebop_maven/data
    ap.set_defaults(do_fit=True, synth_suffix="", model_files=[None])
    args = ap.parse_args()

    synth_test_ds_dir = Path(f"./datasets/synthetic-mist-tess-dataset{args.synth_suffix}/")
    test_results_subdir = f"testing{args.synth_suffix}"

    # This will get the config, labels and published params for formal targets not excluded
    targets_config_file = Path("./config/formal-test-dataset.json")
    formal_targs_cfg = dict(formal_testing.iterate_target_configs(targets_config_file))
    formal_targs = list(formal_targs_cfg.keys())
    #formal_targs = np.array(["V436 Per", "CM Dra"])

    for file_counter, model_file in enumerate(args.model_files, 1):
        print(f"\nModel file {file_counter} of {len(args.model_files)}: {model_file}\n")

        # Set up the estimator and the reporting directory for this model
        if model_file is None or model_file.parent.name == "estimator": # published with ebop_maven
            result_dir = Path("./drop/training/published/" + test_results_subdir)
            model_file = Path("./ebop_maven/data/estimator/default-model.keras")
        else:
            result_dir = model_file.parent / test_results_subdir
        result_dir.mkdir(parents=True, exist_ok=True)

        def warnings_to_stdout(message, category, filename, lineno, file=None, line=None):
            """ Will redirect any warning output to stdout where it can be picked up by Tee """
            sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))
        warnings.showwarning = warnings_to_stdout

        labs, all_preds = None, {}
        with redirect_stdout(Tee(open(result_dir / "model_testing.log", "w", encoding="utf8"))):
            print(f"\nStarting tests on {model_file.parent.name} @",
                  f"{datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
            print("\nEvaluation test dataset:   ", synth_test_ds_dir)
            print("Results will be written to:", result_dir)
            print("\nRuntime environment:", sys.prefix.replace("'", ""))
            print(*(f"{lib.__name__} v{lib.__version__}" for lib in [tf, keras]), sep="\n")
            print(f"tensorflow sees {len(tf.config.list_physical_devices('GPU'))} physical GPU(s)")

            # Report on the basic performance of the model/Estimator predictions vs labels
            for pred_type, iters, dataset_dir, targs in [
                    ("nonmc",   1,      synth_test_ds_dir,          None),
                    # Resource hog & non-essential; takes ~0.5 h on i7 CPU and may not fit in GPU
                    #("mc",      1000,   synth_mist_test_ds_dir,     None),

                    ("nonmc",   1,      FORMAL_TEST_DATASET_DIR,    formal_targs),
                    ("mc",      1000,   FORMAL_TEST_DATASET_DIR,    formal_targs),
            ]:
                print(f"\nEvaluating the model's {pred_type} estimates (iters={iters})",
                      f"on {dataset_dir.name}\n" + "="*80)
                evaluate_model_against_dataset(model_file, iters, targs,
                                               dataset_dir, result_dir / "eval")

            if args.do_fit:
                # In depth report on fitting the formal-test-dataset based on estimator predictions.
                # First loop uses labels as "predictions" to yield a set of control fit results for
                # use as a comparison baseline for subsequent fitted values from model predictions.
                ctrl_fit_vals = None # To be set on the first, control fit run
                for (pred_type, is_ctrl_fit, iterations) in [
                        ("control",     True,       0),
                        ("nonmc",       False,      1),
                        ("mc",          False,      1000),
                ]:
                    print(f"\nTesting JKTEBOP fitting of {pred_type} input values\n" + "="*80)
                    fitted_vals = fit_formal_test_dataset(model_file,
                                                        formal_targs_cfg,
                                                        formal_targs,
                                                        iterations,
                                                        True,
                                                        is_ctrl_fit,
                                                        None if is_ctrl_fit else ctrl_fit_vals,
                                                        result_dir / "fit")
                    if is_ctrl_fit:
                        ctrl_fit_vals = fitted_vals
            else:
                print("The fitting tests have been suppressed by command line switch")

            print(f"\nCompleted tests @ {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
