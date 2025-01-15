""" Script for generating the formal testing dataset of real systems. """
from typing import Iterable
from pathlib import Path
from inspect import getsourcefile
from contextlib import redirect_stdout
import json
from timeit import default_timer
from datetime import timedelta, date

import numpy as np

# pylint: disable=no-member
import astropy.units as u
import tensorflow as tf

from deblib import orbital
from ebop_maven import deb_example

from traininglib import datasets, formal_testing, param_sets, pipeline, plots
from traininglib.tee import Tee

# In this case there us no need to generate any intermediate CSV files
# as the system parameters are already known and are held this config file.
targets_config_file = Path("./config/formal-test-dataset.json")
dataset_dir = Path("./datasets/formal-test-dataset/")
dataset_dir.mkdir(parents=True, exist_ok=True)


def make_formal_test_dataset(config_file: Path,
                             output_dir: Path,
                             target_names: Iterable[str]=None,
                             save_param_csv: bool=True,
                             verbose: bool=True,
                             simulate: bool=True) -> Path:
    """
    This creates a dataset based on real systems; their TESS light curve data with derived features
    & labels from published works. The information required to carry this out is supplied as a json
    file in the input_file argument. The following example shows one target from the input file and
    the tags that are used in this function. Not all tags are mandatory; mission defaults to "TESS",
    author to "SPOC", exptime to "short", "quality_bitmask" to "default", "flux_column" to
    "sap_flux", ecc and omega are assumed to be zero if omitted. bP will be calculated from other
    values if omitted.

    Example config:
    {
        "V436 Per": {
            "mission": "TESS" | "HLPSP",
            "author": "SPOC" | "TESS-SPOC",
            "exptime": "long" | "short" | "fast" | int (s),
            "sectors": {
                "18": {
                    "quality_bitmask": "hardest" | "hard" | "default",
                    "flux_column": "sap_flux" | "pdcsap_flux",
                    "primary_epoch": 1813.201149,
                    "period": 25.935953,
                    "ecc": 0.3835,
                    "omega": 109.56,
                    "labels": {
                        "rA_plus_rB": 0.08015,
                        "k": 1.097,
                        "bP": 0.59,
                        "inc": 87.951,
                        "ecosw": -0.12838,
                        "esinw": 0.3614,
                        "J": 1.041,
                        "L3": -0.003
                    }
                }
            }
        }
    }

    :input_file: the input json file containing the parameters for one or more targets
    :output_dir: the directory to write the output dataset tfrecord file
    :target_names: a list of targets to select from input_file, or None for all
    :save_param_csv: whether to create a csv of the labels/features added to datasets
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    :returns: the Path of the newly created dataset file
    """
    # pylint: disable=invalid-name, too-many-locals, too-many-branches, too-many-statements
    start_time = default_timer()

    if verbose:
        print(f"""
Build formal test dataset based on downloaded lightcurves from TESS targets.
----------------------------------------------------------------------------
The input configuration files is:   {config_file}
Output dataset will be written to:  {output_dir}
Selected targets are:               {', '.join(target_names) if target_names else 'all'}\n""")
        if simulate:
            print("Simulate requested so no dataset will be written, however fits are cached.\n")

    with open(config_file, mode="r", encoding="utf8") as f:
        targets = json.load(f)
        if target_names:
            targets = { name: targets[name] for name in target_names if name in targets }

    out_file = output_dir / f"{config_file.stem}.tfrecord"
    csv_file = output_dir / f"{config_file.stem}.csv"
    if not simulate:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ds = tf.io.TFRecordWriter(f"{out_file}", datasets.ds_options)
        csv_file.unlink(missing_ok=True)

    try:
        inst_counter = 0
        csv_dicts = []
        for target_counter, (target, target_cfg) in enumerate(targets.items(), start=1):
            if verbose:
                print(f"\nProcessing target {target_counter} of {len(targets)}: {target}")

            # Open and concat all of the sector lightcurves for this target
            (lc, _) = formal_testing.prepare_lightcurve_for_target(target, target_cfg, verbose)

            # These are mandatory, so error if missing
            labels = target_cfg["labels"]
            pe = pipeline.to_lc_time(target_cfg["primary_epoch"], lc)
            period = target_cfg["period"] * u.d

            # Produce multiple mags set (varying #bins) available for serialization
            if verbose:
                print(f"{target}: Creating phase normalized, folded lightcurves about",
                        f"{pe.format} {pe} & {period}.")
            mags_features = {}
            for mag_name, mags_bins in deb_example.stored_mags_features.items():
                # Phase folding the light-curve, then interpolate for the mags features
                # Make sure the normalized fold has the primary/phase-zero at index 0 (like JKTEBOP)
                wrap_phase = u.Quantity(1.0)
                fold_lc = lc.fold(period, pe, wrap_phase=wrap_phase, normalize_phase=True)
                _, mags = pipeline.get_sampled_phase_mags_data(fold_lc, mags_bins, wrap_phase)
                mags_features[mag_name] = mags

            # ecc is not used as a label but is needed to calculate phiS and impact params
            ecosw, esinw = labels["ecosw"], labels["esinw"]
            ecc = np.sqrt(np.add(np.power(ecosw, 2), np.power(esinw, 2)))

            # May need to calculate sini and cosi if not present
            inc_rad = np.deg2rad(labels["inc"])
            labels.setdefault("sini", np.sin(inc_rad))
            labels.setdefault("cosi", np.cos(inc_rad))

            # May need to calculate the primary impact parameter label as it's rarely published.
            bP = labels.get("bP", None)
            if bP is None:
                rA = np.divide(labels["rA_plus_rB"], np.add(1, labels["k"]))
                labels["bP"] = orbital.impact_parameter(rA, labels["inc"], ecc, esinw)
                if verbose:
                    print(f"{target}: No impact parameter (bP) supplied;",
                            f"calculated rA = {rA} and then bP = {labels['bP']}")

            # Now assemble the extra features needed: phiS (phase secondary) and dS_over_dP
            extra_features = {
                "phiS": orbital.phase_of_secondary_eclipse(ecosw, ecc),
                "dS_over_dP": orbital.ratio_of_eclipse_duration(esinw)
            }

            # Serialize the labels, folded mags (lc) & extra_features as a deb_example and write
            if not simulate:
                if verbose:
                    print(f"{target}: Saving serialized instance to dataset:", out_file)
                ds.write(deb_example.serialize(target, labels, mags_features, extra_features))
            elif verbose:
                print(f"{target}: Simulated saving serialized instance to dataset:", out_file)
            inst_counter += 1

            # Will be written out to file below
            csv_dicts.append({ "target": target, **labels, **extra_features })
    finally:
        if ds:
            ds.close()
        if save_param_csv and len(csv_dicts) > 0:
            param_sets.write_to_csv(csv_file, csv_dicts, append=False)

    action = "Finished " + ("simulating the saving of" if simulate else "saving")
    print(f"\n{action} {inst_counter} instance(s) from {len(targets)} target(s) to", out_file)
    print(f"The time taken was {timedelta(0, round(default_timer()-start_time))}.")
    return out_file


def write_targets_tabular_file(targets_cfg: dict, targets_tex_file: Path, cite_numeric: bool=True):
    """
    Will write out a tex tabular block containing the table and layout for the
    targets set up in the passed configuration dictionary, in the order they're listed.

    :targets_cfg: the configs for each of the targets
    :targets_tex_file: the file to (over)write.
    :cite_numeric: whether to make reference citations to numeric via 'citealias'
    """

    this_module = Path(getsourcefile(lambda:0)).name
    with open(targets_tex_file, mode="w", encoding="utf8") as tex:
        tex.write(f"% *** Block generated by {this_module} on {date.today()} ***\n")

        # Get the distinct references in the correct order
        ref_list = []
        for _, config in targets_cfg.items():
            for ref in [r.strip() for r in config.get("reference", "").split(",")]:
                if len(ref) > 0 and ref not in ref_list:
                    ref_list.append(ref)

        # We can use numbered cite aliases to get space-saving numeric citations
        if cite_numeric:
            tex.writelines(f"\\defcitealias{{{r}}}{{{i}}} " for i, r in enumerate(ref_list, 1))
            tex.write("\n")

        # Start the tabular and write the header rows
        tex.writelines([
            "\\begin{tabular}{lrrrrrrrccccl}\n",
            "\\hline\n",
            " & $r_{\\rm A}+r_{\\rm B}$ & $k$ & $J$ & ",
                "$e\\cos{\\omega}$ & $e\\sin{\\omega}$ & $i~(\\degr)$ & ",
                "$L_{\\rm 3}$ & Spectral Type & \\multicolumn{3}{|c|}{Flags} & Reference \\\\ \n",
            " & & & & & & & & & T & E & C & \\\\ \n",
            "\\hline\n"
        ])

        for target, config in targets_cfg.items():
            sectors = formal_testing.list_sectors_in_target_config(config)
            label = config.get("label", None) or target
            labels = config["labels"] # Mandatory, so error if missing

            # Write out a line to the tex tabular
            ref = config.get("reference", None)
            cite = f"\\{'citetalias' if cite_numeric else 'cite'}{{{ref}}}" if ref else ""
            tex.write(f"{label} " \
                + f"& {labels['rA_plus_rB']:#.2g} " \
                + f"& {labels['k']:#.2g} " \
                + f"& {labels['J']:#.2g} " \
                + f"& {labels['ecosw']:#.2g} " \
                + f"& {labels['esinw']:#.2g} " \
                + f"& {labels['inc']:#.3g} " \
                + f"& {labels['L3']:#.2g} " \
                + f"& {config['SpT']} " \
                + ("& \\checkmark " if config.get("transits", False) else "& ") \
                + ("& \\checkmark " if config.get("eclipses_similar", False) else "& ") \
                + ("& \\checkmark " if config.get("components_similar", False) else "& ") \
                + f"& {cite} \\\\ % TESS sectors: {', '.join(f'{s}' for s in sectors)}\n")

        # All targets done. Close out the tabular before closing the tex file
        tex.write("\\hline\n\\end{tabular}")
        if cite_numeric:
            # Add a table note with the list of all of the references and their aliases
            tex.writelines([
                "\n",
                "\\\\\n",
                "\\vspace{1ex}\n",
                "{\\raggedright \\textbf{Notes.} \\\\\n",
	            "\\emph{Flags:} we indicate whether the system shows transits (T), ",
                  "has similar eclipse depths (E) or has similar sized components (C) \\\\\n",
                "\\emph{Reference:}",
                  *[f"{',' if i>1 else ''} ({i}) \\cite{{{r}}}" for i, r in enumerate(ref_list, 1)],
                ".\\par\n",
                "}\n"
                "% *** Generated block ends ***"
            ])

# ------------------------------------------------------------------------------
# Makes the formal test dataset of real systems with TESS photometry.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Details of only those targets not excluded from testing
    targets_config = dict(formal_testing.iterate_target_configs(targets_config_file,
                                                                include_excluded=False))
    inc_targs = list(targets_config.keys())

    with redirect_stdout(Tee(open(dataset_dir / "dataset.log", "w", encoding="utf8"))):
        # Process differs from that with synthetic data.
        # We have a config file with MAST search params & labels (from published works).
        # Build the dataset directly by downloading fits & folding LCs.
        # This will include all targets, including those excluded.
        ds_file = make_formal_test_dataset(config_file=targets_config_file,
                                           output_dir=dataset_dir,
                                           target_names=inc_targs,
                                           save_param_csv=True,
                                           verbose=True,
                                           simulate=False)

    # Plot a H-R diagram of those test targets not excluded from testing
    fig = plots.plot_formal_test_dataset_hr_diagram(targets_config)
    fig.savefig(dataset_dir / "formal-test-hr-logl-vs-logteff.pdf")
    fig.clf()

    # The mags features of these targets not marked as excluded from testing
    fig = plots.plot_dataset_instance_mags_features([ds_file], inc_targs, cols=5)
    fig.savefig(dataset_dir / "formal-test-mags-features.pdf")
    fig.savefig(dataset_dir / "formal-test-mags-features.png", dpi=150)
    fig.clf()

    write_targets_tabular_file(targets_config, dataset_dir / "formal-test-targets-tabular.tex")
