""" Helper functions specific to this project. """
from typing import List, Dict, Generator
from pathlib import Path
import csv

def write_param_sets_to_csv(file_name: Path,
                            param_sets: List[Dict],
                            field_names: List[any] = None) -> None:
    """
    Writes the list of parameter set dictionaries to a csv file.

    :file_name: the full name of the file to create or overwrite.
    :param_sets: the list of dictionaries to write out.
    :field_names: the list of fields to write, in the required order. If
    None, the field_names will be read from the first item in param_sets
    """
    # This data is saved in an intermediate form as we've yet to
    # generate the actual light-curves. We use csv, as this is
    # easily read/consumed by apps for reviewing and the
    # tensorflow data API for subsequent processing.
    if field_names is None:
        field_names = param_sets[0].keys()
    with open(file_name, mode="w", encoding="UTF8", newline='') as f:
        dw = csv.DictWriter(f,
                            field_names,
                            quotechar="'",
                            quoting=csv.QUOTE_NONNUMERIC)
        dw.writeheader()
        dw.writerows(param_sets)

def read_param_sets_from_csv(file_name: Path) -> Generator[Dict, any, None]:
    """
    Reads a list of parameter set dictionaries from a csv file,
    as created by write_param_sets_to_csv()

    :file_name: the full name of the csv file containing the parameter sets
    :returns: a generator for the dictionaries
    """
    with open(file_name, mode="r", encoding="UTF8") as pf:
        dr = csv.DictReader(pf, quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
        for param_set in dr:
            yield param_set

def read_param_sets_from_csvs(file_names: List[Path]) -> Generator[Dict, any, None]:
    """
    Reads a list of parameter set dictionaries from across all of the csv files,
    as created by write_param_sets_to_csv()

    :file_names: the full names of the csv files containing the parameter sets
    :returns: a generator for the dictionaries
    """
    for file_name in file_names:
        for param_set in read_param_sets_from_csv(file_name):
            yield param_set
