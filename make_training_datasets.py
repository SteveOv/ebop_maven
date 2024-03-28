""" Script for generating the training datasets. """
import os
from pathlib import Path
from contextlib import redirect_stdout

from ebop_maven import trainsets
from ebop_maven.libs.tee import Tee

datasets_dir = Path("./datasets")

# Tell the libraries where the JKTEBOP executable lives.
# The conda yaml based env sets this but it's not set for venvs.
if not "JKTEBOP_DIR" in os.environ:
    os.environ["JKTEBOP_DIR"] = "~/jktebop43"

# Creating a training dataset is a two step process;
#   1. create a set of trainset csv files
#   2. from these build a corresponding set of tensorflow (tfrecord) dataset files
#
# The reason for the split are;
#   - it allows for the mulitple datasets to be created from a consistent set of
#     instances (the trainset). This is useful during training/hyperparameter tuning
#     by allow what's in a dataset to be varied (e.g. phase shift of the lightcurves)
#   - tfrecord files are not easily readable so the csv files are useful for
#     access to the original parameters of an instance
#   - it's a convenient break in the process
#
# Trainsets
# ---------
# A trainset is a set of one or more csv files where each row is a set of
# the basic parameters for a single instance, minus the folded lightcurve
# (model) data. For each instance, the parameters are a superset of those
# required to generate the corresponding folded lightcurve, the training labels,
# and other data which may be useful for troubleshooting. The CSV format makes
# it easy to read in either a text editor or spreadsheet application.
#
# Trainsets need to cover a range of potential system configurations and there
# are two strategies for generating them;
#   1. random selection of "lightcurve" parameter values by sampling
#      appropriate distributions for each parameter
#   2. using a grid of physical configurations combined with stellar models
#      to generate systems from which the lightcurve parameters are derived
#
# The former approach is used for training data and the latter is useful for testing.
#
# Datasets
# --------
# A dataset builds on a trainset, primarily with the addition of a folded,
# model lightcurve. These are written to a set of TensorFlow tfrecord files
# which mirror the csv files of the trainset. Within the tfrecord files, each
# "row" holds the data for an instance and consists of;
#   - features (which the ML model learns to interpret)
#       - the folded lightcurve
#       - additional single feature values which augment the ligthcurve
#   - labels (the ground truth used in the training process)
#       - these are the values we want the ML model to predict
#

# The formal-trainset will be used for training the final model
# First we create the readable csv trainset from random parameter distributions.
trainset_dir = datasets_dir / "formal-trainset"
trainset_dir.mkdir(parents=True, exist_ok=True)
with redirect_stdout(Tee(open(trainset_dir/"trainset.log", "w", encoding="utf8"))):
    trainsets.write_trainset_from_distributions(100000, 10, trainset_dir, verbose=True)

# TODO: generate the tensorflow dataset from the trainset
