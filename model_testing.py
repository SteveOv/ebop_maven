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

import astropy.units as u
import numpy as np
from keras import Model

from ebop_maven.libs.tee import Tee
from ebop_maven.libs import deb_example, lightcurve, jktebop, stellar, limb_darkening, orbital
from ebop_maven.estimator import Estimator
from ebop_maven import datasets
import plots

def test_model_against_formal_test_dataset(
        use_estimator: Union[Model, Estimator],
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
    :returns: a tuple of List[predictions dicts], List[labels dicts],
    one row per target instance in the order of the dataset
    """
    # Create our Estimator. It will tell us what its inputs should look like
    # and which labels it (& the underlying model) can predict.
    if not isinstance(use_estimator, Estimator):
        use_estimator = Estimator(use_estimator)

    # Gets the target ids (names), labels and mags/ext features to predict on
    ids, labels, features = get_dataset_labels_and_features(
                test_dataset_dir,
                label_names=[n for n in use_estimator.prediction_names if not n.endswith("sigma")],
                feature_names=use_estimator.input_feature_names,
                mags_bins=use_estimator.mags_feature_bins,
                mags_wrap_phase=use_estimator.mags_feature_wrap_phase,
                scaled_labels=scaled,
                include_ids=include_ids)
    if include_ids is not None:
        assert len(include_ids) == len(ids)

    # Make our prediction which will return [{"name": value, "name_sigma": sigma_value}]*insts
    print(f"\nThe Estimator is making predictions on the {len(ids)} formal test instances",
          f"with {mc_iterations} iteration(s) (iterations >1 triggers MC Dropout algorithm).")
    predictions = use_estimator.predict(features, mc_iterations, unscale=not scaled)

    # Echo some summary statistics
    prediction_type = "mc" if mc_iterations > 1 else "nonmc"
    (_, _, _, ocs) = get_label_and_prediction_raw_values(labels, predictions)
    print("\n--------------------------------")
    print(f"Total MAE ({prediction_type}): {np.mean(np.abs(ocs)):.9f}")
    print(f"Total MSE ({prediction_type}): {np.mean(np.power(ocs, 2)):.9f}")
    print("--------------------------------\n")
    return (np.array(predictions), np.array(labels))


def fit_against_formal_test_dataset(
        input_params: List[Dict[str, float]],
        labels: List[Dict[str, float]],
        targets_config: Dict[str, any],
        selected_targets: List[str]=None,
        apply_fit_overrides: bool=True) -> np.ndarray[Dict[str, float]]:
    """
    Will fit members of the formal test dataset, as configured in targets_config,
    based on the sets of input_params passed in returning the corresponding fitted params.

    It's important that input_params, labels and selected_targets (or targets_config keys)
    are all of the same length and in the same order.

    :input_params: the list of params Dicts to fit with, one per target
    :labels: the equivalent list of labels Dicts, for reporting
    :targets_config: the full config for all targets
    :selected_targets: list of target ids to fit, or all if empty
    :apply_fit_overrides: apply any fit_overrides from each target's config
    :returns: a List of the targets fitted parameter Dicts
    """
    tw = TextWrapper(100)
    fitted_params = []
    if selected_targets is None or len(selected_targets) == 0:
        selected_targets = list(targets_config.keys())
    assert len(selected_targets) == len(input_params)
    assert len(selected_targets) == len(labels)
    l_names = list(input_params[0].keys())

    for ix, (target, target_input_params, target_labels) in enumerate(zip(selected_targets,
                                                                          input_params,
                                                                          labels)):
        target_cfg = targets_config[target]
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
        table_of_predictions_vs_labels([target_labels], [target_input_params], [target])

        # Handle the estimator predicting bP rather than inc directly
        if not "inc" in target_input_params and "bP" in target_input_params:
            inc = calculate_inc_from_other_predictions(target_input_params)
            print(f"Input param calculated from other predictions; inc = {inc:.6f}")
            target_input_params["inc"] = inc.to(u.deg).value

        # published fitting params that may be needed for good fit
        fit_overrides = target_cfg.get("fit_overrides", {}) if apply_fit_overrides else {}
        lrats = fit_overrides.pop("lrat", [])

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
        fit_params = jktebop.read_fitted_params_from_par_lines(par_contents, l_names)

        print(f"\nHave fitted {target} resulting in the following fitted params")
        table_of_predictions_vs_labels([target_labels], [fit_params], [target])
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


def calculate_inc_from_other_predictions(preds: Dict[str, float]) -> u.deg:
    """
    Calculate inc from the impact parameter and other supporting prediced values.

    :preds: the prediction dictionary for this instance
    :returns: the calculated inc (as a Quantity in deg)
    """
    required_preds = ["k", "rA_plus_rB", "bP", "esinw", "ecosw"]
    missing_preds= [k for k in required_preds if k not in preds]
    if missing_preds:
        raise KeyError("These required predicted values are missing:", ", ".join(missing_preds))

    b = preds["bP"]
    r = preds["rA_plus_rB"] / (1 + preds["k"])
    esinw = preds["esinw"]
    e = np.sqrt(np.add(np.power(preds["ecosw"], 2), np.power(esinw, 2)))
    return orbital.orbital_inclination(r, b, e, esinw, orbital.EclipseType.PRIMARY)


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


def table_of_predictions_vs_labels(
        labels: List[Dict[str, float]],
        predictions: List[Union[Dict[str, float], Dict[str, Tuple[float, float]]]],
        block_headings: List[str],
        selected_label_names: List[str]=None,
        reverse_scaling: bool=False,
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

    line_len = 13 + (11 * len(keys))-1 + 22
    for rhead, rlabs, rpreds, rocs in zip(block_headings, raw_labels, pred_noms, ocs):
        # Plot a sub table for each row of labels/predictions/ocs
        to.write("-"*line_len + "\n")
        to.write(f"{rhead:<10s} | " + " ".join(f"{k:>10s}" for k in keys + ["MAE", "MSE"]))
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
    for (targ, lrow, mrow, frow) in datasets.inspect_dataset(tfrecord_files, include_ids,
                                                             scale_labels=scaled_labels):
        ids += [targ]
        ds_labels += [{ ln: lrow[ln] for ln in label_names}]
        ds_features += [{
            "mags": mrow[deb_example.create_mags_key(mags_bins, mags_wrap_phase)], 
            **{ fn: frow[fn] for fn in feature_names if fn not in ["mags"] }}
        ]
    return ids, ds_labels, ds_features

def get_label_and_prediction_raw_values(
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
    if selected_labels is None:
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
            errors = [[pd.get(f"{l}_sigma", 0) for l in selected_labels] for pd in predictions]
            errors = np.array(errors)
        else:
            errors = np.zeros_like(nominals)

    # Coalesce any None values to zero otherwise we'll get failures below
    # pylint: disable=singleton-comparison
    label_values[label_values == None] = 0.
    nominals[nominals == None] = 0.
    errors[errors == None] = 0.
    # pylint: enable=singleton-comparison

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
    estimator = Estimator(Path("./drop/cnn_ext_model.keras"))
    trainset_name = estimator.metadata["trainset_name"]
    mags_key = f"mags_{estimator.mags_feature_bins}_{estimator.mags_feature_wrap_phase}"
    save_dir = Path(f"./drop/results/{estimator.name}/{trainset_name}/{mags_key}")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open("./config/formal-test-dataset.json", mode="r", encoding="utf8") as tf:
        targets_cfg = json.load(tf)

    exclude_targets = ["V402 Lac", "V456 Cyg"] # Neither are suitable for JKTEBOP fitting
    targets = np.array([target for target in targets_cfg if target not in exclude_targets])
    trans_flags = np.array([targets_cfg.get(t, {}).get("transits", False) for t in targets])
    all_labels, all_preds, all_fits = None, {}, {}

    with redirect_stdout(Tee(open(save_dir / "model_testing.log", "w", encoding="utf8"))):
        print(fill(f"Testing {estimator.name} against targets: {', '.join(targets)}", 100))

        # Report on the performance of the model/Estimator predictions vs labels
        for ptype, iters in [("nonmc", 1), ("mc", 1000)]:
            print(f"\n\nTesting the model's {ptype} estimates (where iters={iters})\n" + "="*80)
            (all_preds[ptype], all_labels) = test_model_against_formal_test_dataset(
                                                                        estimator, iters, targets)

            results_stem = "predictions_vs_labels_" + ptype # pylint: disable=invalid-name
            fig = plots.plot_predictions_vs_labels(all_labels, all_preds[ptype], trans_flags)
            fig.savefig(save_dir / f"{results_stem}.eps")

            with open(save_dir / f"{results_stem}.csv", mode="w", encoding="utf8") as cf:
                predictions_vs_labels_to_csv(all_labels, all_preds[ptype], targets, to=cf)

            with open(save_dir / f"{results_stem}.txt", mode="w", encoding="utf8") as tf:
                for (heading, mask) in [("All targets", [True]*len(targets)),
                                        ("\n\nTransiting systems only", trans_flags),
                                        ("\n\nNon-transiting systems only", ~trans_flags)]:
                    tf.write(f"\n{heading}\n")
                    table_of_predictions_vs_labels(
                                    all_labels[mask], all_preds[ptype][mask], targets[mask], to=tf)

        # Now report using the predictions as input to fitting the format-test-dataset with JKTEBOP
        # First we add a control item - this allows us to fit against the labels to get control fits
        all_preds["control"] = copy.deepcopy(all_labels)
        for ptype in ["control", "nonmc", "mc"]:
            print(f"\n\nTesting JKTEBOP fitting based on the model's {ptype} estimates\n" + "="*80)
            all_fits[ptype] = fit_against_formal_test_dataset(
                                        all_preds[ptype], all_labels, targets_cfg, targets, True)

            results_stem = "fitted_params_vs_labels_" + ptype # pylint: disable=invalid-name
            fig = plots.plot_predictions_vs_labels(all_labels, all_fits[ptype], trans_flags)
            fig.savefig(save_dir / f"{results_stem}.eps", dpi=300)

            with open(save_dir / f"{results_stem}.txt", "w", encoding="utf8") as of:
                for (heading, mask) in [("All targets", [True]*len(targets)),
                                        ("\n\nTransiting systems only", trans_flags),
                                        ("\n\nNon-transiting systems only", ~trans_flags)]:
                    of.write(f"\n{heading}\n")
                    table_of_predictions_vs_labels(
                                    all_labels[mask], all_fits[ptype][mask], targets[mask], to=of)
