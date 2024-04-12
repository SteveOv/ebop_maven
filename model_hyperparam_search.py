"""
Searches for the best set of hyperparams for the Mags/Extra-Features model
"""
from typing import Callable, Union
from pathlib import Path
from contextlib import redirect_stdout
import os
import random as python_random
import json

import numpy as np
import tensorflow as tf
import keras

from keras import layers, optimizers, callbacks as cb
from tensorflow.python.framework.errors_impl import OpError # pylint: disable=no-name-in-module

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

from ebop_maven.libs import deb_example
from ebop_maven import modelling
from ebop_maven.libs.tee import Tee

TRAINSET_NAME = "formal-training-dataset/"
DATASET_DIR = Path(".") / "datasets" / TRAINSET_NAME
FORMAL_TESTSET_DIR = Path(".") / "datasets/formal-test-dataset/"

MODEL_FILE_NAME = "parameter-search-model"

MAX_HYPEROPT_EVALS = 250        # Maximum number of distinct Hyperopt evals to run
TRAINING_EPOCHS = 100           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
EARLY_STOPPING_PATIENCE = 10    # Number of epochs w/o improvement before stopping

ENFORCE_REPEATABILITY = True    # If true, avoid GPU/CUDA cores for repeatable results
SEED = 42                       # Standard random seed ensures repeatable randomization
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

CALLBACKS = [
    cb.EarlyStopping("val_loss", restore_best_weights=True,
                     patience=EARLY_STOPPING_PATIENCE, verbose=1)
]

print("\n".join(f"{lib.__name__} v{lib.__version__}" for lib in [tf, keras]))
if ENFORCE_REPEATABILITY:
    # Extreme, but it stops TensorFlow/Keras from using (even seeing) the GPU.
    # Slows training down massively (by 3-4 times) but should avoid GPU memory
    # constraints! Necessary if repeatable results are required (Keras advises
    # that out of order processing within GPU/CUDA can lead to varying results).
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(f"Found {len(tf.config.list_physical_devices('GPU'))} GPU(s)\n")


# -----------------------------------------------------------
# Set up the training/validation/test datasets
# -----------------------------------------------------------
print(f"\nPicking up training/validation/test datasets within '{DATASET_DIR}'",
      f"& the formal testing dataset (real data) from within '{FORMAL_TESTSET_DIR}'.")
ds_titles = ["training", "validation", "testing", "formal testing"]
datasets = [tf.data.TFRecordDataset] * len(ds_titles)
ds_counts = [int] * len(ds_titles)
map_func = deb_example.create_map_func(noise_stddev=lambda: 0.005,
                                       roll_steps=lambda: tf.random.uniform([], -9, 10, tf.int32))
for ds_ix, (label, set_dir) in enumerate([("training", DATASET_DIR),
                                          ("validation", DATASET_DIR),
                                          ("testing", DATASET_DIR),
                                          (None, FORMAL_TESTSET_DIR)]):
    # Don't set up any mappings/shuffle/batch yet as we want to count the contents first
    if ds_ix < 3:
        files = list(set_dir.glob(f"**/{label}/**/*.tfrecord"))
        if ds_ix == 0:
            (datasets[ds_ix], ds_counts[ds_ix]) = \
                deb_example.create_dataset_pipeline(files, BATCH_FRACTION, map_func,
                                                    shuffle=True, reshuffle_each_iteration=True,
                                                    max_buffer_size=MAX_BUFFER_SIZE,
                                                    prefetch=1, seed=SEED)
        else:
            (datasets[ds_ix], ds_counts[ds_ix]) = \
                deb_example.create_dataset_pipeline(files, BATCH_FRACTION, map_func)
    else:
        # For the formal test dataset simple pipeline with no noise/roll and a single batch
        files = list(set_dir.glob("**/*.tfrecord"))
        (datasets[ds_ix], ds_counts[ds_ix]) = deb_example.create_dataset_pipeline(files, 10000)

    print(f"Found {ds_counts[ds_ix]:,} {label} instances spread over",
          f"{len(files)} tfrecord file(s) within '{set_dir}'.")


# -----------------------------------------------------------
# Temporary model Helpers (will go when modelling updated)
# -----------------------------------------------------------
def dnn_with_taper(num_layers: int,
                   units: int,
                   kernel_initializer: any,
                   activation: any,
                   dropout_rate: float=0,
                   taper_units: int=0) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ Creates the function to build the requested DNN layers """
    def layers_func(prev_tensor: keras.KerasTensor) -> keras.KerasTensor:
        prev_tensor = modelling.hidden_layers(prev_tensor, num_layers, units, kernel_initializer,
                                              activation, dropout_rate,
                                              name_prefix=("Hidden-", "Dropout-"))
        if taper_units:
            prev_tensor = modelling.hidden_layers(prev_tensor, 1, taper_units, kernel_initializer,
                                                  activation, name_prefix=("Taper-", ))
        return prev_tensor
    return layers_func

def simple_symmetric_cnn(num_layers: int,
                         filters: int,
                         kernel_size: int,
                         strides: int,
                         padding: str,
                         activation: str) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ Creates the function to build the requested DNN layers """
    def layers_func(prev_tensor: keras.KerasTensor) -> keras.KerasTensor:
        prev_tensor = modelling.conv1d_layers(prev_tensor, num_layers=num_layers, filters=filters,
                                              kernel_size=kernel_size, strides=strides,
                                              padding=padding, activation=activation,
                                              name_prefix="CNN-")
        return prev_tensor
    return layers_func

def empty_layer() -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ Creates an empty passthrough layer """
    def layers_func(prev_tensor: keras.KerasTensor) -> keras.KerasTensor:
        return modelling.empty_layer(prev_tensor)
    return layers_func

def output_layer(units: int=8,
                 kernel_initializer:str = "glorot_uniform",
                 activation: str = "linear") -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ Create an output layer """
    def layers_func(prev_tensor: keras.KerasTensor) -> keras.KerasTensor:
        return modelling.output_layer(prev_tensor, units=units,
                                      kernel_initializer=kernel_initializer, activation=activation)
    return layers_func


# -----------------------------------------------------------
# Define the target model and the hyperparameter space
# -----------------------------------------------------------
#pylint: disable=line-too-long
scope.define(layers.ReLU)
scope.define(layers.LeakyReLU)
scope.define(layers.PReLU)
scope.define(optimizers.Adam)
scope.define(optimizers.Nadam)

trials_pspace = hp.choice("train_and_test_model", [{
    "model": hp.choice("model", [{
        "func": modelling.build_mags_ext_model,
        "build_mags_layers": hp.choice("build_mags_layers", [
            {
                "func": simple_symmetric_cnn,
                "num_layers": hp.choice("cnn_num_layers", [4, 5, 6]),
                "filters": hp.choice("cnn_filters", [32, 64, 96]),
                "kernel_size": hp.choice("cnn_kernel_size", [16, 8]),
                "strides": hp.choice("cnn_strides", [4, 2]),
                "padding": hp.choice("cnn_padding", ["same"]),
                "activation": hp.choice("cnn_activation", ["relu"])
            },
        ]),
        "build_ext_layers": {
            "func": empty_layer
        },
        "build_dnn_layers": hp.choice("build_dnn_layers", [
            {
                "func": dnn_with_taper,
                "num_layers": hp.choice("dnn_num_layers", [1, 2, 3]),
                "units": hp.choice("dnn_units", [64, 128, 256]),
                "kernel_initializer": hp.choice("dnn_init", ["glorot_uniform", "he_uniform", "he_normal"]),
                "activation": hp.choice("activation", ["relu", "leaky_relu"]),
                "dropout_rate": hp.choice("dnn_dropout", [0.3, 0.4, 0.5, 0.6]),
                "taper_units": hp.choice("dnn_taper", [None, 32, 64, 128]),
            },
        ]),
        "build_output_layer": {
            "func": output_layer
        },
    }]),

    "optimizer": hp.choice("optimizer", [
        { "class": optimizers.Adam, "learning_rate": hp.choice("adam_lr", [1e-5, 5e-5, 1e-6]) },
        { "class": optimizers.Nadam, "learning_rate": hp.choice("nadam_lr", [1e-5, 5e-5, 1e-6]) }
    ]),

    "loss_function": hp.choice("loss_function", ["mae", "mse"]),      
}])


def get_trial_value(trial_dict: dict, key: str, pop_it: bool=False) -> Union[Callable, any]:
    """
    Will get the requested value from the trial dictionary. Specifically handles the special
    case where we are getting a function/class with hp.choices over the args by parsing the
    function and its kwargs list and then executing it.

    Example of get the "model" which is the result of the build_mags_ext_model() function
    with the accompanying kwargs. Also, handles that build_dnn_layers kwarg is a nested function.

    "model": {
        "func": "<function build_mags_ext_model at 0x7d0270610d60>"
        "build_dnn_layers": {
          "func": "<function dnn_with_taper at 0x7d0270543920>",
          "activation": "leaky_rely",
          ...
        },
        "build_ext_layers": {
          "func": "<function empty_layer at 0x7d0270610360>"
        },
        ...
    """
    # We want a KeyError if item not found
    target_value = trial_dict.pop(key) if pop_it else trial_dict.get(key)

    # We're looking for the special case: a dict with a func/class item and the rest the kwargs.
    if isinstance(target_value, dict) and ("func" in target_value or "class" in target_value):
        the_callable = target_value.get("func", target_value.get("class"))
        if isinstance(the_callable, str): # support it being a str (easier to read when reporting)
            the_callable = eval(the_callable) # pylint: disable=eval-used
        callable_kwargs = {}
        for kwarg in target_value: # recurse to handle funcs which have funcs as args
            if kwarg not in ["func", "class"]:
                callable_kwargs[kwarg] = get_trial_value(target_value, kwarg)
        return the_callable(**callable_kwargs)
    return target_value


# -----------------------------------------------------------
# Conduct the trials
# -----------------------------------------------------------
def train_and_test_model(trial_kwargs):
    """
    Evaluate a single set of hyperparams by building, training and evaluating a model on them.
    """
    print("\nEvaluating model and hyperparameters based on the following trial_kwargs:\n"
          + json.dumps(trial_kwargs, indent=4, sort_keys=False, default=str))

    loss = candidate = history = None
    status = STATUS_FAIL

    optimizer = get_trial_value(trial_kwargs, "optimizer")
    loss_function = get_trial_value(trial_kwargs, "loss_function")
    fixed_metrics = ["mae", "mse"]

    try:
        # Build and Compile the trial model
        # always use the same metrics as we use them for trial evaluation
        candidate = get_trial_value(trial_kwargs, "model", False)
        candidate.compile(optimizer=optimizer, loss=[loss_function], metrics=fixed_metrics)

        print(f"\nTraining the following model against {ds_counts[0]} {ds_titles[0]} instances.")
        print(candidate.summary(line_length=120, show_trainable=True))
        history = candidate.fit(x = datasets[0],
                                epochs = TRAINING_EPOCHS,
                                callbacks = CALLBACKS,
                                validation_data = datasets[1],
                                verbose=2)

        print(f"\nEvaluating model against {ds_counts[2]} {ds_titles[2]} dataset test instances.")
        candidate.evaluate(x=datasets[2], y=None, verbose=2)

        print(f"\nFull evaluation against {ds_counts[3]} {ds_titles[3]} dataset instances.")
        results = candidate.evaluate(x=datasets[3], y=None, verbose=2)

        # Always MAE from metrics, as this allows us to vary the loss function
        # during training while having a consistent metric for trial evaluation.
        loss = results[1 + fixed_metrics.index("mae")]
        status = STATUS_OK
    except OpError as exc:
        print(f"*** Training failed! *** Caught a {type(exc).__name__}: {exc.op} / {exc.message}")
        print(f"The problem hyperparam set is: {trial_kwargs}\n")

    return { "loss": loss, "model": candidate, "history": history, "status": status }

# Conduct the trials
results_dir = Path(".") / "drop" / "hyperparam_search"
results_dir.mkdir(parents=True, exist_ok=True)
with redirect_stdout(Tee(open(results_dir / "search.log", "w", encoding="utf8"))):
    trials = Trials()
    best = fmin(fn = train_and_test_model,
                space = trials_pspace,
                trials = trials,
                algo = tpe.suggest,
                max_evals = MAX_HYPEROPT_EVALS,
                loss_threshold = 0.04,
                catch_eval_exceptions = True)


# -----------------------------------------------------------
# Report on the outcome
# -----------------------------------------------------------
best_model = trials.best_trial["result"]["model"]
best_params = space_eval(trials_pspace, best)
print("\nBest model hyperparameter set is:\n"
      + json.dumps(best_params, indent=4, sort_keys=False, default=str))

# Save the best model / best parameter set
modelling.save_model(results_dir / "best_mode.keras", best_model)
with open(results_dir / "best_params.json", mode="w", encoding="utf8") as of:
    json.dump(best_params, of, indent=4, default=str)
