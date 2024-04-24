"""
Searches for the best set of hyperparams for the Mags/Extra-Features model
"""
# pylint: disable=too-many-arguments
from typing import Callable, Union, List, Dict, Tuple
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
TRAINSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "training"
VALIDSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "validation"
TESTSET_DIR = Path(".") / "datasets" / "synthetic-mist-tess-dataset"
FORMAL_TESTSET_DIR = Path(".") / "datasets/formal-test-dataset/"

MODEL_FILE_NAME = "parameter-search-model"

# The subset of all available labels which we will train to predict
CHOSEN_LABELS = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc"]

MAX_HYPEROPT_EVALS = 250        # Maximum number of distinct Hyperopt evals to run
TRAINING_EPOCHS = 250           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
PATIENCE = 7                    # Number of epochs w/o improvement before stopping

ENFORCE_REPEATABILITY = True    # If true, avoid GPU/CUDA cores for repeatable results
SEED = 42                       # Standard random seed ensures repeatable randomization
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)


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
ds_titles = ["training", "validation", "testing", "formal testing"]
ds_dirs = [TRAINSET_DIR, VALIDSET_DIR, TESTSET_DIR, FORMAL_TESTSET_DIR]
datasets = [tf.data.TFRecordDataset] * len(ds_titles)
counts = [int] * len(ds_titles)
for ds_ix, (label, set_dir) in enumerate(zip(ds_titles, ds_dirs)):
    files = list(set_dir.glob("**/*.tfrecord"))
    if ds_ix < 3:
        map_func = deb_example.create_map_func(labels=CHOSEN_LABELS,
                                        noise_stddev=lambda: 0.005,
                                        roll_steps=lambda: tf.random.uniform([], -9, 10, tf.int32))
        if ds_ix == 0:
            (datasets[ds_ix], counts[ds_ix]) = \
                deb_example.create_dataset_pipeline(files, BATCH_FRACTION, map_func,
                                                    shuffle=True, reshuffle_each_iteration=True,
                                                    max_buffer_size=MAX_BUFFER_SIZE,
                                                    prefetch=1, seed=SEED)
        else:
            (datasets[ds_ix], counts[ds_ix]) = \
                deb_example.create_dataset_pipeline(files, BATCH_FRACTION, map_func)
    else:
        # For the formal test dataset simple pipeline with no noise/roll and a single batch
        map_func = deb_example.create_map_func(labels=CHOSEN_LABELS) # No added noise or roll
        datasets[ds_ix], counts[ds_ix] = deb_example.create_dataset_pipeline(files, 10000, map_func)
    print(f"Found {counts[ds_ix]:,} {label} instances over {len(files)} tfrecord files in", set_dir)


# -----------------------------------------------------------
# Temporary model Helpers (will go when modelling updated)
# -----------------------------------------------------------
def cnn_with_pooling(num_layers: int=4,
                     filters: Union[int, List[int]]=64,
                     kernel_size: Union[int, List[int]]=8,
                     strides: Union[int, List[int]]=2,
                     padding: str="same",
                     activation: str="relu",
                     pooling_ixs: Union[int, List[int]]=None,
                     pooling_type: layers.Layer=None,
                     pooling_kwargs: Union[Dict, List[Dict]]=None):
    """
    Prototype of creating a set of CNN layers with optional pooling at given indices.
    """
    if not isinstance(filters, List):
        filters = [filters] * num_layers
    if not isinstance(kernel_size, List):
        kernel_size = [kernel_size] * num_layers
    if not isinstance(strides, List):
        strides = [strides] * num_layers
    if pooling_kwargs is None:
        pooling_kwargs = { "pool_size": 2, "strides": 2 }
    if pooling_ixs and pooling_type and pooling_kwargs and isinstance(pooling_kwargs, dict):
        num_pools = len(pooling_ixs)
        pooling_kwargs = [pooling_kwargs] * num_pools
    def layers_func(input_tensor: keras.KerasTensor) -> keras.KerasTensor:
        # Expected failure if any list isn't num_layers long
        pooling_ix = 0
        for cnn_ix in range(num_layers):
            if pooling_ixs and pooling_type and cnn_ix+pooling_ix in pooling_ixs:
                input_tensor = pooling_type(name=f"Pool-{pooling_ix+1}",
                                            **pooling_kwargs[pooling_ix])(input_tensor)
                pooling_ix += 1

            input_tensor = layers.Conv1D(filters=filters[cnn_ix],
                                         kernel_size=kernel_size[cnn_ix],
                                         strides=strides[cnn_ix],
                                         padding=padding,
                                         activation=activation,
                                         name=f"CNN-{cnn_ix+1}")(input_tensor)
        return input_tensor
    return layers_func

def cnn_scaled_pairs_with_pooling(num_pairs: int=2,
                                  filters: int=32,
                                  kernel_size: int=16,
                                  strides: int=None,
                                  scaling_multiplier: int=2,
                                  padding: str="same",
                                  activation: str="relu",
                                  pooling_type: layers.Layer=None,
                                  pooling_kwargs: Union[Dict, List[Dict]]=None,
                                  trailing_pool: bool=True):
    """
    Pairs of Conv1d layers where the filters & kernel_size/strides can
    optionally be scaled up/down for each successive pair (by scaling_multiplier).
    Each pair can optionally be followed with a pooling layer. If we are
    including pooling layers the trailing_pool flag dictates whether the
    pooling layer after the final pair of Conv1ds is appended or not.
    """
    if pooling_kwargs is None:
        pooling_kwargs = { "pool_size": 2, "strides": 2 }

    def layer_func(input_tensor: keras.KerasTensor) -> keras.KerasTensor:
        this_filters = filters
        this_kernel_size = kernel_size
        this_strides = strides if strides else max(1, this_kernel_size // 2)

        for ix in range(num_pairs):
            for sub_ix in range(2):
                input_tensor = layers.Conv1D(filters=this_filters,
                                             kernel_size=this_kernel_size,
                                             strides=this_strides,
                                             padding=padding,
                                             activation=activation,
                                             name=f"CNN-{ix+1}-{sub_ix+1}")(input_tensor)

            if pooling_type and (trailing_pool or ix < num_pairs-1):
                input_tensor = pooling_type(name=f"Pool-{ix+1}", **pooling_kwargs)(input_tensor)

            if scaling_multiplier != 1:
                this_filters *= scaling_multiplier
                this_kernel_size = max(1, this_kernel_size // scaling_multiplier)
                if strides:
                    this_strides = max(1, this_strides // scaling_multiplier)
                else:
                    this_strides = max(1, this_kernel_size // 2)
        return input_tensor
    return layer_func

def cnn_fixed_pairs_with_pooling(num_pairs: int=2,
                                 filters: int=64,
                                 kernel_size: int=4,
                                 strides: int=None,
                                 padding: str="same",
                                 activation: str="relu",
                                 pooling_type: layers.Layer=None,
                                 pooling_kwargs: Union[Dict, List[Dict]]=None,
                                 trailing_pool: bool=True):
    """
    Pairs of Conv1d layers with fixed filters, kernel_size and strided and
    optionally followed with a pooling layer.
    """
    return cnn_scaled_pairs_with_pooling(num_pairs, filters, kernel_size, strides, 1,
                                         padding, activation,
                                         pooling_type, pooling_kwargs, trailing_pool)

def dnn_with_taper(num_layers: int,
                   units: int,
                   kernel_initializer: any,
                   activation: any,
                   dropout_rate: float=0,
                   taper_units: int=0) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ Creates the function to build the requested DNN layers """
    def layers_func(prev_tensor: keras.KerasTensor) -> keras.KerasTensor:
        prev_tensor = modelling.hidden_layers(num_layers, units, kernel_initializer,
                                              activation, dropout_rate,
                                              name_prefix=("Hidden-", "Dropout-"))(prev_tensor)
        if taper_units:
            prev_tensor = modelling.hidden_layers(1, taper_units, kernel_initializer,
                                                  activation, name_prefix=("Taper-", ))(prev_tensor)
        return prev_tensor
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

# Shared choices
cnn_padding_choice = hp.choice("cnn_padding", ["same", "valid"])
cnn_activation_choice = hp.choice("cnn_activation", ["relu"])
cnn_pooling_type_choice = hp.choice("cnn_pooling_type", [layers.AvgPool1D, layers.MaxPool1D, None])
cnn_trailing_pool_choice = hp.choice("cnn_trailing_pool", [True, False])
dnn_kernel_initializer_choice = hp.choice("dnn_init", ["he_uniform", "he_normal", "glorot_uniform"])
learning_rate_choice = hp.choice("learning_rate", [1e-5, 5e-5, 1e-4])

trials_pspace = hp.choice("train_and_test_model", [{
    "model": hp.choice("model", [{
        "func": modelling.build_mags_ext_model,
        "mags_layers": hp.choice("mags_layers", [
            {
                # Current CNN from best performing model from manual search
                "func": cnn_fixed_pairs_with_pooling,
                "num_pairs": 4,
                "filters": 64,
                "kernel_size": 4,
                "strides": 2,
                "padding": "same",
                "activation": "relu",
                "pooling_type": layers.MaxPool1D,
                "pooling_kwargs": { "pool_size": 2, "strides": 2 },
                "trailing_pool": False,
            },
            {
                # Pairs of Conv1ds with fixed filters/kernels/strides and optional pooling layers
                "func": cnn_fixed_pairs_with_pooling,
                "num_pairs": hp.choice("cnn_fixed_num_layers", [3, 4]),
                "filters": hp.choice("cnn_fixed_filters", [32, 64]),
                "kernel_size": hp.choice("cnn_fixed_kernel_size", [4, 8]),
                "strides": None, # Always kernel_size/2
                "padding": cnn_padding_choice,
                "activation": cnn_activation_choice,
                "pooling_type": cnn_pooling_type_choice,
                "trailing_pool": cnn_trailing_pool_choice,
            },
            {
                # Pairs of Conv1ds with doubling filters & halving kernels/strides per pair
                # and optional pooling layers
                "func": cnn_scaled_pairs_with_pooling,
                "num_pairs": hp.choice("cnn_scaled_num_layers", [3]),
                "filters": hp.choice("cnn_scaled_filters", [16, 32]),
                "kernel_size": hp.choice("cnn_scaled_kernel_size", [16, 8]),
                "strides": hp.choice("cnn_scaled_strides", [2]),
                "scaling_multiplier": 2,
                "padding": "same",
                "activation": cnn_activation_choice,
                "pooling_type": cnn_pooling_type_choice,
                "trailing_pool": cnn_trailing_pool_choice,
            },
            {
                # Randomized CNN with/without pooling.
                "func": cnn_with_pooling,
                "num_layers": hp.choice("cnn_num_layers", [4, 5, 6, 7]),
                "filters": hp.choice("cnn_filters", [32, 64]),
                "kernel_size": hp.choice("cnn_kernel_size", [16, 8]),
                "strides": hp.choice("cnn_strides", [4, 2]),
                "padding": cnn_padding_choice,
                "activation": cnn_activation_choice,
                "pooling_ixs": hp.choice("cnn_pooling_ixs", [None, [2], [2, 5]]),
                "pooling_type": cnn_pooling_type_choice,
            },
        ]),
        "ext_layers": None,
        "dnn_layers": hp.choice("dnn_layers", [
            {
                "func": dnn_with_taper,
                "num_layers": hp.choice("dnn_num_layers", [1, 2, 3]),
                "units": hp.choice("dnn_units", [64, 128, 256]),
                "kernel_initializer": dnn_kernel_initializer_choice,
                "activation": hp.choice("activation", ["relu", "leaky_relu", "elu"]),
                "dropout_rate": hp.choice("dnn_dropout", [0.4, 0.5, 0.6]),
                "taper_units": hp.choice("dnn_taper", [None, 32, 64]),
            },
        ]),
        "output": {
            "func": modelling.output_layer,
            "label_names_and_scales": { l: deb_example.labels_and_scales[l] for l in CHOSEN_LABELS },
            "kernel_initializer": dnn_kernel_initializer_choice,
            "activation": "linear"
        },
    }]),

    "optimizer": hp.choice("optimizer", [
        { "class": optimizers.Adam, "learning_rate": learning_rate_choice },
        { "class": optimizers.Nadam, "learning_rate": learning_rate_choice }
    ]),

    "loss_function": hp.choice("loss_function", ["mae", "mse", "huber"]),      
}])


def get_trial_value(trial_dict: dict,
                    key: str,
                    pop_it: bool=False, 
                    tuples_to_lists: bool=True) -> Union[Callable, any]:
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

    # Workaround for the nasty behaviour in hyperopt where lists get silently converted to tuples
    # (see: https://github.com/hyperopt/hyperopt/issues/526)
    if isinstance(target_value, Tuple) and tuples_to_lists:
        target_value = list(target_value)

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
    print("\n" + "-"*80,
          "Evaluating model and hyperparameters based on the following trial_kwargs:",
          json.dumps(trial_kwargs, indent=4, sort_keys=False, default=str))

    weighted_loss = candidate = history = None
    status = STATUS_FAIL

    optimizer = get_trial_value(trial_kwargs, "optimizer")
    loss_function = get_trial_value(trial_kwargs, "loss_function")
    fixed_metrics = ["mae", "mse", "r2_score"]

    try:
        # Build and Compile the trial model
        # always use the same metrics as we use them for trial evaluation
        candidate = get_trial_value(trial_kwargs, "model", False)
        candidate.compile(optimizer=optimizer, loss=[loss_function], metrics=fixed_metrics)

        # Reset the tf random seed so shuffling & other "random" behaviour is repeated
        tf.random.set_seed(SEED)

        print(f"\nTraining the following model against {counts[0]} {ds_titles[0]} instances.")
        print(candidate.summary(line_length=120, show_trainable=True))
        history = candidate.fit(x = datasets[0],
                                epochs = TRAINING_EPOCHS,
                                callbacks = [cb.EarlyStopping("val_loss", restore_best_weights=True,
                                                              patience=PATIENCE, verbose=1)],
                                validation_data = datasets[1],
                                verbose=2)

        print(f"\nEvaluating model against {counts[2]} {ds_titles[2]} dataset test instances.")
        candidate.evaluate(x=datasets[2], y=None, verbose=2)

        print(f"\nFull evaluation against {counts[3]} {ds_titles[3]} dataset instances.")
        results = candidate.evaluate(x=datasets[3], y=None, verbose=2)

        # Out final loss is always MAE from metrics. This allows us to vary the
        # training loss function while using a consistent metric for trial evaluation.
        mae = results[1 + fixed_metrics.index("mae")]
        mse = results[1 + fixed_metrics.index("mse")]

        # The trial is evaluated on a "weighted loss"; the loss modified with a penalty
        # on model complexity/#params (which is approximated from the number of trainable params).
        weights = int(sum(np.prod(s) for s in [w.shape for w in candidate.trainable_weights]))
        params = np.log(weights)
        weighted_loss = mse * params
        status = STATUS_OK
        print("-"*80,
             f"Trial result: MAE = {mae:.6f}, MSE = {mse:.6f} & params(ln[weights]) = {params:.6f}",
             f"{' '*14}giving a weighted loss(mse*params) = {weighted_loss:6f}")

        features = candidate.get_layer("Mags-Input").output.shape[1] \
                    + candidate.get_layer("Ext-Input").output.shape[1]
        aic = features*np.log(mse) + 2*params
        bic = features*np.log(mse) + np.log(features)*params
        print(f"Alternatively: AIC = {aic:,.3f} and BIC = {bic:,.3f}", "-"*80)
    except OpError as exc:
        print(f"*** Training failed! *** Caught a {type(exc).__name__}: {exc.op} / {exc.message}")
        print(f"The problem hyperparam set is: {trial_kwargs}\n")

    return { "loss": mae, "status": status, "mae": mae, "mse": mse,
            "weighted_loss": weighted_loss, "AIC": aic, "BIC": bic, "model": candidate, "history": history }

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
