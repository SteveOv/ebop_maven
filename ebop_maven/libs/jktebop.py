""" Module for interacting with the JKTEBOP dEB light-curve fitting tool. """
# pylint: disable=invalid-name
from typing import Union, Dict, List, Callable
import os
import threading
import subprocess
import tempfile
from io import TextIOBase
from inspect import getsourcefile
from pathlib import Path
from string import Template

import numpy as np
from lightkurve import LightCurve
from astropy.io import ascii as io_ascii

_this_dir = Path(getsourcefile(lambda:0)).parent
_template_files = {
    2: _this_dir / "data/jktebop/task2.in.template",
    3: _this_dir / "data/jktebop/task3.in.template"
}

_jktebop_directory = \
    Path(os.environ.get("JKTEBOP_DIR", "~/jktebop/")).expanduser().absolute()


def get_jktebop_dir() -> Path:
    """
    Publishes the path of the directory holding the JKTEBOP executable
    """
    return _jktebop_directory


def run_jktebop_task(in_filename: Path,
                     out_filename: Path=None,
                     delete_files_pattern: str=None,
                     stdout_capture: Union[TextIOBase, Callable[[str], None]]=None):
    """
    Will run JKTEBOP against the passed in file, waiting for the production of the
    expected out file. The contents of the outfile will be returned line by line.

    The function returns a Generator[str] yielding the lines of text writteing to
    the out_filename. These can be read in a for loop:
    ```Python
    for line in run_jktebop_task(...):
        ...
    ```
    or, to capture everything wrap it in a list() function:
    ```Python
    lines = list(run_jktebop_task(...))
    ```

    To use the stdout_capture functionality you can pass in an instance of a
    TextIOBase, for example:
    ```Python
    output=io.StringIO()
    run_jktebop_task(..., stdout_capture=output)
    ```
    or, you can pass in a function or lambda which takes a single str argument
    which will be called each time the console is written to. For example the
    print() function will work to echo JKTEBOP's stdout to the current console:
    ```Python
    run_jktebop_task(..., stdout_capture=print)
    ```

    :in_filename: the path of the in file containing the JKTEBOP input parameters.
    :out_filename: the path of the primary output file we expect be created.
    This will be read and the contents yielded in a Generator.
    :delete_files_pattern: optional glob pattern of files to be deleted after
    successful processing. The files will not be deleted if there is a failure.
    :stdout_capture: a callback or TextIO given each line JKTEBOP writtes to stdout as it happens
    :returns: yields the content of the primary output file, line by line.
    """
    # Call out to jktebop to process the in file and generate the requested output file
    return_code = None
    cmd = ["./jktebop", f"{in_filename.name}"]
    with subprocess.Popen(cmd,
                          cwd=get_jktebop_dir(),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          text=True) as proc:
        # This is an async call so we can publish any writes to stdout/stderr as they happen
        stdout_thread = None
        if stdout_capture is not None:
            def echo_to_capture_console():
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if isinstance(stdout_capture, TextIOBase):
                        print(line.strip(), file=stdout_capture)
                    elif isinstance(stdout_capture, Callable):
                        stdout_capture(line.strip())
            stdout_thread = threading.Thread(target=echo_to_capture_console)
            stdout_thread.start()
        return_code = proc.wait()
        if stdout_thread:
            stdout_thread.join()

    # JKTEBOP (v43) doesn't appear to set the response code on failures so
    # we'll check if there has been a problem by trying to pick up the out file.
    if return_code == 0 and out_filename.exists():
        # Read the resulting out file
        with open(out_filename, mode="r", encoding="utf8") as of:
            for line in of:
                yield line
    elif not out_filename.exists():
        raise subprocess.CalledProcessError(0, cmd)

    if delete_files_pattern:
        # Optional cleanup
        for parent in [in_filename.parent, out_filename.parent]:
            for file in parent.glob(delete_files_pattern):
                file.unlink()


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
    with tempfile.NamedTemporaryFile(dir=get_jktebop_dir(),
                                     prefix=file_prefix,
                                     suffix=".in",
                                     delete=False,
                                     mode="w",
                                     encoding="utf8") as wf:
        in_filename = Path(wf.name)
        out_filename = in_filename.parent / (in_filename.stem + ".out")
        in_params["out_filename"] = f"{out_filename.name}"
        write_in_file(in_filename, 2, **in_params)

    # Call out to jktebop to process the in file & generate the corresponding .out file
    # with the modelled LC data, which we parse and return as shape [2, #rows]
    return np.loadtxt(run_jktebop_task(in_filename, out_filename, f"{in_filename.stem}.*"),
                      usecols=(0, 1), comments="#", dtype=np.double, unpack=True)


def write_in_file(file_name: Path,
                  task: int,
                  append_lines: List[str]=None,
                  **params):
    """
    Writes a JKTEBOP .in file based on applying the passed params/token
    values to the template file corresponding to the selected task.

    :file_name: the name and path of the file to write.
    :task: the task being undertaken. Currently only 2 and 3 are supported.
    :append_lines: lines to optionally append at the end of the in file.
    :params: a dictionary of param tokens/keys and values.
    """
    if file_name is None or not isinstance(file_name, Path):
        raise TypeError("file_name is not a Path")

    # Pre-process the params/tokens to be applied to the .in file.
    in_params = _prepare_params_for_task(task, params)

    if "L3" in in_params and in_params["L3"] < 0.:
        raise ValueError("Minimum L3 input value is 0.0")
    if "rA_plus_rB" in in_params and in_params["rA_plus_rB"] > 0.8:
        raise ValueError("Maximum rA_plus_rB input value is 0.8")

    if "file_name_stem" not in in_params:
        in_params["file_name_stem"] = file_name.stem

    with open(file_name, mode="w", encoding="utf8") as wf:
        with open(_template_files[task], "r", encoding="utf8") as tpf:
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
    if not isinstance(task, int):
        raise TypeError("task must be an int")
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
