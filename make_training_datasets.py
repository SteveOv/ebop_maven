""" Script for generating the training datasets. """
import os
from pathlib import Path
from contextlib import redirect_stdout

from ebop_maven import trainsets, datasets, deb_example
from ebop_maven.libs.tee import Tee

RESUME = False
config_dir = Path("./config")
datasets_root = Path("./datasets")

# Tell the libraries where the JKTEBOP executable lives.
# The conda yaml based env sets this but it's not set for venvs.
if not "JKTEBOP_DIR" in os.environ:
    os.environ["JKTEBOP_DIR"] = "~/jktebop/"

# Creating a training dataset is a two step process;
#   1. create a set of trainset csv files
#   2. from these build a corresponding set of tensorflow (tfrecord) dataset files
#
# The reason for the split are;
#   - it allows for the mulitple datasets to be created from a consistent set of
#     instances (the trainset). This is useful during training/hyperparameter tuning
#     by allow what's in a dataset to be varied (e.g. phase shift of the lightcurves)
#   - tfrecord files are not easily readable so the csv files are useful for
#     access to the original parameters of an instance. You could consider the
#     dataset as the compiled output with the trainset being the source
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
# A dataset builds on a trainset, primarily with the addition of a folded, binned
# model lightcurve. These are written to a set of TensorFlow tfrecord files
# which mirror the csv files of the trainset. Within the tfrecord files, each
# "row" holds the data for an instance and consists of;
#   - features (which the ML model learns to interpret)
#       - the folded lightcurve (now in a variety of configurations)
#       - additional single feature values which augment the lightcurve
#   - labels (the ground truth used in the training process)
#       - these are the values we want the ML model to predict
#


# ------------------------------------------------------------------------------
# The formal training dataset based on sampling parameter distributions
# ------------------------------------------------------------------------------
dataset_dir = datasets_root / "formal-training-dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)
with redirect_stdout(Tee(open(dataset_dir / "trainset.log", "w", encoding="utf8"))):
    trainsets.write_trainset(instance_count=250000,
                             file_count=25,
                             output_dir=dataset_dir,
                             generator_func=trainsets.generate_instances_from_distributions,
                             verbose=True,
                             simulate=False)

trainsets.plot_trainset_histograms(dataset_dir, dataset_dir / "train-histogram-full.png", cols=3)
trainsets.plot_trainset_histograms(dataset_dir, dataset_dir / "train-histogram-main.eps", cols=2,
                                    params=["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"])

with redirect_stdout(Tee(open(dataset_dir/"dataset.log", "a" if RESUME else "w", encoding="utf8"))):
    datasets.make_dataset_files(trainset_files=sorted(dataset_dir.glob("trainset*.csv")),
                                output_dir=dataset_dir,
                                valid_ratio=0.2,
                                test_ratio=0,
                                resume=RESUME,
                                max_workers=5,
                                verbose=True,
                                simulate=False)

# ------------------------------------------------------------------------------
# A second testing dataset based on MIST models and a configured parameter space
# ------------------------------------------------------------------------------
dataset_dir = datasets_root / "synthetic-mist-tess-dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)
with redirect_stdout(Tee(open(dataset_dir / "trainset.log", "w", encoding="utf8"))):
    trainsets.write_trainset(instance_count=20000,
                             file_count=10,
                             output_dir=dataset_dir,
                             generator_func=trainsets.generate_instances_from_mist_models,
                             verbose=True,
                             simulate=False)

trainsets.plot_trainset_histograms(dataset_dir, dataset_dir / "synth-histogram-full.png", cols=3)
trainsets.plot_trainset_histograms(dataset_dir, dataset_dir / "synth-histogram-main.eps", cols=2,
                                    params=["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"])

with redirect_stdout(Tee(open(dataset_dir/"dataset.log", "a" if RESUME else "w", encoding="utf8"))):
    datasets.make_dataset_files(trainset_files=sorted(dataset_dir.glob("trainset*.csv")),
                                output_dir=dataset_dir,
                                valid_ratio=0.,
                                test_ratio=1.,
                                resume=RESUME,
                                max_workers=5,
                                verbose=True,
                                simulate=False)


# ------------------------------------------------------------------------------
# The final, formal test dataset based on real TESS lightcurves of known systems
# ------------------------------------------------------------------------------
targets_config_file = config_dir / "formal-test-dataset.json"
dataset_dir = datasets_root / "formal-test-dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)
with redirect_stdout(Tee(open(dataset_dir / "dataset.log", "w", encoding="utf8"))):
    # Process differs from that with synthetic data. We have a config file with MAST search params
    # & labels (from published works). Build the dataset directly by downloading fits & folding LCs.
    formal_testset_file = datasets.make_formal_test_dataset(config_file=targets_config_file,
                                                            output_dir=dataset_dir,
                                                            target_names=None,
                                                            verbose=True,
                                                            simulate=False)
