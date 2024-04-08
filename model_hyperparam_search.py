"""
Searches for the best set of hyperparams for the Mags/Extra-Features model
"""
from pathlib import Path
import os
import random as python_random

import numpy as np
import tensorflow as tf
import keras

from keras import layers, optimizers, callbacks as cb
from tensorflow.python.framework.errors_impl import OpError # pylint: disable=no-name-in-module

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

from ebop_maven.libs import deb_example
from ebop_maven import modelling

TRAINSET_NAME = "formal-training-dataset/"
DATASET_DIR = Path(".") / "datasets" / TRAINSET_NAME
FORMAL_TESTSET_DIR = Path(".") / "datasets/formal-test-dataset/"

MODEL_FILE_NAME = "parameter-search-model"

TRAINING_EPOCHS = 100           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
EARLY_STOPPING_PATIENCE = 10    # Number of epochs w/o improvement before stopping
MAX_HYPEROPT_EVALS = 100        # Maximum number of distinct Hyperopt evals to run

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
# Define the target model and the hyperparameter space
# -----------------------------------------------------------
scope.define(layers.ReLU)
scope.define(layers.LeakyReLU)
scope.define(layers.PReLU)
scope.define(optimizers.Adam)
scope.define(optimizers.Nadam)
hyperparam_space = hp.choice("model_fn", [
        {
            "model_fn": modelling.build_mags_ext_model,
            # Parameters
            "build_mags_layers": lambda lt: modelling.conv1d_layers(lt, 5, 64, 8, 4),
            #"build_ext_layers": lambda lt: modelling.empty_layer(lt),
            "build_dnn_layers": lambda lt: modelling.hidden_layers(lt, 2, 256, "he_normal", "leaky_relu", 0.5),
            #"build_output_layer": lambda lt: modelling.output_layer(lt),

            # Compile() parameters
            "optimizer": hp.choice("optimizer", [optimizers.Adam, optimizers.Nadam]),
            "learning_rate": hp.choice("learning_rate", [5e-5, 1e-5, 1e-6]),
            "loss_function": hp.choice("loss_function", ["mae", "mse"]),
        },
    ])



# -----------------------------------------------------------
# Conduct the trials
# -----------------------------------------------------------
def evaluate_hyperparams(model_kwargs):
    """
    Evaluate a single set of hyperparams by building, training and evaluating a model on them.
    """

    model_fn = model_kwargs.pop("model_fn", None)
    print(f"\nEvaluating {model_fn.__name__} with the set of hyperparameters.\n{model_kwargs}")

    loss = candidate = history = None
    status = STATUS_FAIL
    fixed_metrics = ["mae", "mse"]
    try:
        learning_rate = model_kwargs.pop("learning_rate", None)
        optimizer = model_kwargs.pop("optimizer", None)
        loss_function = model_kwargs.pop("loss_function", None)

        candidate = model_fn(**model_kwargs)

        # Compile it - always use the same metrics as we use them for trial evaluation
        candidate.compile(optimizer=optimizer(learning_rate),
                          loss=[loss_function],
                          metrics=fixed_metrics)

        print(f"Training the following model against {ds_counts[0]} {ds_titles[0]} instances.")
        print(candidate.summary())
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
        print(f"The problem hyperparam set is: {model_kwargs}\n")

    return { "loss": loss, "model": candidate, "history": history, "status": status }

dataset_name = f"{DATASET_DIR.parent.name}/{DATASET_DIR.name}"
output_dir = Path(f"./saves/results/hyperparams_search/{dataset_name}")

# Conduct the trials
trials = Trials()
best = fmin(fn = evaluate_hyperparams,
            space = hyperparam_space,
            trials = trials,
            algo = tpe.suggest,
            max_evals = MAX_HYPEROPT_EVALS,
            loss_threshold = 0.04,
            catch_eval_exceptions = True)


# -----------------------------------------------------------
# Report on the outcome
# -----------------------------------------------------------
best_model = trials.best_trial["result"]["model"]
best_params = space_eval(hyperparam_space, best)

# TODO: Save best model


# TODO: Save best params as JSON
print(f"\nBest model hyperparameter set is: {best_params}")
