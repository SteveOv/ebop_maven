""" Module for interacting with the JKTEBOP dEB light-curve fitting tool. """
# pylint: disable=invalid-name
from typing import Union, Dict, List, Callable
import os
import subprocess
import tempfile
from inspect import getsourcefile
from pathlib import Path
from string import Template

import numpy as np
from lightkurve import LightCurve
from astropy.io import ascii as io_ascii

_this_dir = Path(getsourcefile(lambda:0)).parent
_task2_template_file = _this_dir / "data/jktebop/task2.in.template"
_task3_template_file = _this_dir / "data/jktebop/task3.in.template"
_jktebop_directory = \
    Path(os.environ.get("JKTEBOP_DIR", "~/jktebop43")).expanduser().absolute()


def generate_model_light_curve(file_prefix: str, **params) -> np.ndarray:
    """
    Use JKTEBOP task 2 to generate a model light-curve for the passed
    parameter set. The model data will be returned as an array of 
    shape (2, rows) with column 0 the phase values and column 1 the magnitudes.

    :file_prefix: the prefix to give temp files written jktebop
    :params: a **kwargs dictionary of the system params and values to model.
    See data/jktebop/task2.in.template for the params/tokens used.
    :returns: model data as a shape(2, rows) ndarray
    """
    # Pre-process the params/tokens to be applied to the .in file.
    in_params = _prepare_params_for_task(2, params, calc_refl_coeffs=True)

    # Create a unique temp .in file, for jktebop to process. Set it to write to
    # an output file with an equivalent name so they're both easy to clean up.
    with tempfile.NamedTemporaryFile(dir=_jktebop_directory,
                                     prefix=file_prefix,
                                     suffix=".in",
                                     delete=False,
                                     mode="w",
                                     encoding="utf8") as wf:
        in_filename = Path(wf.name)
        out_filename = in_filename.parent / (in_filename.stem + ".out")
        with open(_task2_template_file, "r", encoding="utf8") as tpf:
            template = Template(tpf.read())

        in_params["out_filename"] = f"{out_filename.name}"
        wf.write(template.substitute(**in_params))

    # Call out to jktebop to process the in file and generate the
    # corresponding .out file with the modelled LC data
    cmd = ["./jktebop", f"{in_filename.name}"]
    result = subprocess.run(cmd,
                            cwd=_jktebop_directory,
                            capture_output=True,
                            check=True)

    # JKTEBOP (v43) doesn't appear to set the response code on failures so
    # we'll check if there has been a problem by trying to pick up the out file.
    model_data = None
    if result.returncode == 0 and out_filename.exists():
        # Read the resulting out file
        model_data = np.loadtxt(fname=out_filename,
                                usecols=(0, 1),
                                comments="#",
                                dtype=np.double,
                                unpack=True)

        # Delete the in and out files
        in_filename.unlink()
        out_filename.unlink()
    elif not out_filename.exists():
        raise subprocess.CalledProcessError(0, cmd, result.stdout)
    return model_data


def write_task3_in_file(file_name: Path,
                        append_lines: List[str]=None,
                        **params):
    """
    Writes a JKTEBOP task3 .in file based on applying the passed params/token
    values to the task3.in.template file.

    :file_name: the name and path of the file to write
    :append_lines: lines to optionally append at the end of the in file
    :params: a dictionary of param tokens/keys and values
    """
    if file_name is None or not isinstance(file_name, Path):
        raise TypeError("file_name is not a Path")

    # Pre-process the params/tokens to be applied to the .in file.
    in_params = _prepare_params_for_task(3, params)

    if "L3" in in_params and in_params["L3"] < 0.:
        raise ValueError("Minimum L3 input value is 0.0")
    if "rA_plus_rB" in in_params and in_params["rA_plus_rB"] > 0.8:
        raise ValueError("Maximum rA_plus_rB input value is 0.8")

    if "file_name_stem" not in in_params:
        in_params["file_name_stem"] = file_name.stem

    with open(file_name, mode="w", encoding="utf8") as wf:
        with open(_task3_template_file, "r", encoding="utf8") as tpf:
            template = Template(tpf.read())

        # Will error if any expected tokens are not present.
        wf.write(template.substitute(**in_params))

        # Add on any lines to be appended to the file
        if append_lines:
            # writelines doesn't put each line on a separate line
            wf.writelines("\n" + l for l in append_lines)


def write_light_curve_to_dat_file(lc: LightCurve,
                                  file_name: Path,
                                  column_names: List[str] = None,
                                  column_formats: List[Union[str, Callable]] = None):
    """
    Will write the time and magnitude columns of the passed LightCurve to a
    JKTEBOP compatible text 'dat' file for use in subsequent fitting.

    :lc: the source LightCurve
    :file_name: the target file which will be overwritten if it already exists
    :column_names: the lc columns to read from [time, delta_mag, delta_mag_err]
    :column_formats: the formats to use for each column on writing them out to
    the file [lambda t: f'{t.jd-2.4e6:.6f}', '%.6f', '%.6f']
    """
    if lc is None:
        raise TypeError("lc is None")
    if file_name is None or not isinstance(file_name, Path):
        raise TypeError("file_name is not a Path")

    if column_names is None:
        column_names = ["time", "delta_mag", "delta_mag_err"]
    if column_formats is None:
        column_formats = [lambda t: f"{t.jd-2.4e6:.6f}", "%.6f", "%.6f"]

    # Check and set up the formats.
    if len(column_names) != len(column_formats):
        raise ValueError("Different number of column_names to column_formats. "
                         + "Each column must have an equivalent format.")

    formats = dict(zip(column_names, column_formats))
    columns = [lc[column_name] for column_name in column_names]
    io_ascii.write(columns,
                   output=file_name,
                   format="no_header",
                   names=column_names,
                   formats=formats,
                   comment="#",
                   delimiter=" ",
                   overwrite=True)


#
# Private helper functions
# pylint: disable=too-many-arguments
def _prepare_params_for_task(task: int,
                             params: dict,
                             fit_rA_and_rB: bool = False,
                             fit_e_and_omega: bool = False,
                             calc_refl_coeffs: bool = False,
                             in_place: bool = False) -> Union[None, Dict]:
    """
    Will prepare the passed params dictionary for subsequent use against a
    JKTEBOP in file template. This function understands the various JKTEBOP
    magic values and such like.

    :task: what task are we targetting
    :params: the dictionary to be prepared
    :fit_rA_rB_directly: if set, JKTEBOP will fit with rA & rB rather than rA+rb & k
    :fit_e_and_omega: if set, JKTEBOP will fit with e & omega rather than ecosw & esinw
    :calc_refl_coeffs: task 2 only - if set, JKTEBOP will calculate reflection coeffs
    :in_place: update the original dictionary (True) or return a copy [False]
    """
    if task is None:
        raise TypeError("task cannot be None")
    if params is None:
        raise TypeError("params cannot be None")

    new_params = params if in_place else params.copy()

    # Apply any defaults for rarely used params
    new_params.setdefault("ring", 5)

    if fit_rA_and_rB:
        # indicate to JKTEBOP to fit/generate for rA and rB, rather than
        # rA+rB and k, by setting rA_plus_rB=-rA and k=rB.
        # Will give KeyErrors if the expected rA and rB values are not present
        new_params["rA_plus_rB"] = np.negative(new_params["rA"])
        new_params["k"] = new_params["rB"]

    if fit_e_and_omega:
        # indicate to JKTEBOP to fit/generate for e and omega, rather than the
        # Poincare elements ecosw and esinw, by setting ecosw=e+10, esinw=omega
        # Will give KeyErrors if the expected e and omega values are not present
        new_params["ecosw"] = np.add(new_params["e"], 10.)
        new_params["esinw"] = new_params["omega"]

    if task == 2 and calc_refl_coeffs:
        # For task 2 JKTEBOP supports calculating the reflected light coeffs.
        # To signal this set the coeffs to a large negative value.
        new_params["reflA"] = -100
        new_params["reflB"] = -100

    if not in_place:
        return new_params
    return None
