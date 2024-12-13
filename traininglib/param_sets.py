"""
Functions for reading and writing training and testing parameter set csv files.
"""
from typing import Generator, Iterable, List
from pathlib import Path
import csv


def write_to_csv(file_name: Path,
                 param_sets: Iterable[dict],
                 field_names: List[any] = None,
                 append: bool=False) -> None:
    """
    Writes the list of parameter set dictionaries to a csv file.

    :file_name: the full name of the file to create or overwrite.
    :param_sets: the list of dictionaries to write out.
    :field_names: the list of fields to write, in the required order. If
    None, the field_names will be read from the first item in param_sets
    :append: whether to append the file exists (True) or to overwrite it (False)
    """
    if field_names is None:
        field_names = param_sets[0].keys()
    with open(file_name, mode="a+" if append else "w", encoding="UTF8", newline='') as f:
        dw = csv.DictWriter(f,
                            field_names,
                            quotechar="'",
                            quoting=csv.QUOTE_NONNUMERIC)
        if not append or not f.tell():
            dw.writeheader()
        dw.writerows(param_sets)


def read_from_csv(file_name: Path) -> Generator[dict, any, None]:
    """
    Reads a list of parameter set dictionaries from a csv file,
    as created by write_param_sets_to_csv()

    :file_name: the full name of the csv file containing the parameter sets
    :returns: a generator for the dictionaries
    """
    with open(file_name, mode="r", encoding="UTF8") as pf:
        yield from csv.DictReader(pf, quotechar="'", quoting=csv.QUOTE_NONNUMERIC)


def read_from_csvs(file_names: Iterable[Path]) -> Generator[dict, any, None]:
    """
    Reads a list of parameter set dictionaries from across all of the csv files,
    as created by write_param_sets_to_csv()

    :file_names: the full names of the csv files containing the parameter sets
    :returns: a generator for the dictionaries
    """
    for file_name in file_names:
        yield from read_from_csv(file_name)


def get_field_names_from_csvs(file_names: Iterable[Path]) -> List[str]:
    """
    Returns a list of the field names common to all of the passed CSV files.

    :file_names: the full names of the csv files containing the parameter sets
    :returns: a list[str] of the common names
    """
    names: list[str] = None
    for file_name in file_names:
        # pylint: disable=not-an-iterable
        with open(file_name, mode="r", encoding="utf8") as pf:
            csv_reader = csv.reader(pf, quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
            this_names = next(csv_reader)

        # Modify names so that it hold the names common to both
        names = [n for n in names if n in this_names] if names is not None else this_names
    return names
