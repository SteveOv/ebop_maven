"""
Searches for the best model/hyperparams/training for the Mags/Extra-Features model
"""
# pylint: disable=too-many-arguments, too-many-locals
from pathlib import Path
from contextlib import redirect_stdout
import os
import sys
import random as python_random
import json
from io import StringIO
from datetime import timedelta
import traceback
from warnings import filterwarnings

import numpy as np
import tensorflow as tf
import keras
from keras import layers, optimizers, callbacks as cb
from tensorflow.python.framework.errors_impl import OpError # pylint: disable=no-name-in-module

from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

from ebop_maven import modelling, deb_example
from ebop_maven.libs.keras_custom.callbacks import TrainingTimeoutCallback
from ebop_maven.libs.tee import Tee

from traininglib import model_search_helpers
import make_trained_cnn_model

# Configure the inputs and outputs of the model
CHOSEN_FEATURES = []
MAGS_BINS = deb_example.default_mags_bins
MAGS_WRAP_PHASE = None # None indicates wrap to centre on midpoint between eclipses
CHOSEN_LABELS = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "bP"]

TRAINSET_NAME = "formal-training-dataset-250k"
TRAINSET_GLOB_TERM = "trainset00?.tfrecord" # Just the first 10 files, so 80k train/20k validation
TRAINSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "training"
VALIDSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "validation"
TESTSET_DIR = Path(".") / "datasets" / "synthetic-mist-tess-dataset"
FORMAL_TESTSET_DIR = Path(".") / "datasets/formal-test-dataset/"

MODEL_FILE_NAME = "search-model"

MAX_HYPEROPT_EVALS = 200            # Maximum number of distinct Hyperopt evals to run
HYPEROPT_LOSS_TH = 0.01             # Will stop search in the unlikely event we get below this loss
TRAINING_EPOCHS = 250               # Set high if we're using early stopping
BATCH_FRACTION = 0.001              # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000          # Size of Dataset shuffle buffer (in instances)
TRAIN_PATIENCE = 7                  # Number of epochs w/o improvement before training is stopped
TRAIN_TIMEOUT = timedelta(hours=1)  # Timeout training if not completed within this time

SEED = 42                           # Standard random seed ensures repeatable randomization
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

results_dir = Path(".") / "drop" / "model_search"
results_dir.mkdir(parents=True, exist_ok=True)

# CUDA_VISIBLE_DEVICES is set for the conda env. A value of "-1" suppresses GPUs which is
# useful for repeatable results (Keras advises that out of order processing within GPU/CUDA
# can lead to varying results) and also avoiding memory constraints on smaller GPUs (mine!).
print("Runtime environment:", sys.prefix.replace("'", ""))
print("\n".join(f"{lib.__name__} v{lib.__version__}" for lib in [tf, keras]))

cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"The environment variable CUDA_VISIBLE_DEVICES set to '{cuda_visible_devices}'")
print(f"Found {len(tf.config.list_physical_devices('GPU'))} GPU(s)\n")


# -----------------------------------------------------------
# Set up the test datasets - we don't need to recreate per trial
# -----------------------------------------------------------
# No added noise or roll as this is already present in the datasets
test_map_func = deb_example.create_map_func(mags_bins=MAGS_BINS, mags_wrap_phase=MAGS_WRAP_PHASE,
                                            ext_features=CHOSEN_FEATURES, labels=CHOSEN_LABELS)
test_ds, test_ct = [tf.data.TFRecordDataset] * 2, [int] * 2
for ti, (tname, tdir) in enumerate(zip(["trial", "real"], [TESTSET_DIR, FORMAL_TESTSET_DIR])):
    tfiles = list(tdir.glob("**/*.tfrecord"))
    (test_ds[ti], test_ct[ti]) = \
        deb_example.create_dataset_pipeline(tfiles, BATCH_FRACTION, test_map_func)
    print(f"Found {test_ct[ti]:,} {tname} test insts in {len(tfiles)} tfrecord files in", tdir)


# -----------------------------------------------------------
# Define the target model and the hyperparameter space
# -----------------------------------------------------------
#pylint: disable=line-too-long
scope.define(layers.ReLU)
scope.define(layers.LeakyReLU)
scope.define(layers.PReLU)
scope.define(optimizers.Adam)
scope.define(optimizers.Nadam)

# Shared choices: define the ranges, choices etc consistently, but keep the hp.*
# func calls specific to model type so fmin/TPE has free reign across each.
cnn_strides_pchoices = [(0.75, None), (0.125, 2), (0.125, 4)]
cnn_strides_fraction_kwargs = { "low": 0.25, "high": 1, "q": 0.25 }
cnn_pooling_type_choices = [layers.AvgPool1D, layers.MaxPool1D, None]
cnn_padding_choices = ["same", "valid"]
cnn_activation_choices = ["relu"]
dnn_initializer_choices = ["he_uniform", "he_normal", "glorot_uniform"]
dnn_activation_choices = ["relu", "leaky_relu", "elu"]
loss_function_choices = [["mae"], ["mse"]] #, ["huber"]]
lr_qlogu_kwargs = { "low": -9, "high": -4.5, "q": 1e-4 } # ~1.1e-2 down to ~1.2e-4
sgd_momentum_uniform_kwargs = { "low": 0.0, "high": 1.0 }
lr_exp_decay_rate_kwargs = { "low": 0.90, "high": 0.95 }
lr_cos_warmup_steps_kwargs = { "low": 0, "high": 3000, "q": 1000 }
lr_cos_decay_steps_kwargs = { "low": 50000, "high": 150000, "q": 10000 }
lr_cos_alpha_choices = [0.01, 0.001]
lr_pw_boundary_choices = [[5000, 30000], [20000, 40000]]
lr_pw_value_choices = [[0.001, 0.0001, 0.00001], [0.0005, 0.00005, 0.00001], [0.0001, 0.001, 0.0001]]

# Genuinely shared choice between DNN layers and output layer
dnn_kernel_initializer_choice = hp.choice("dnn_init", dnn_initializer_choices)

# This will augment the model, giving an Estimator context information
metadata = {
    "extra_features_and_defaults": 
                {f: deb_example.extra_features_and_defaults[f] for f in CHOSEN_FEATURES },
    "mags_bins": MAGS_BINS,
    "mags_wrap_phase": MAGS_WRAP_PHASE,
    "labels_and_scales": {l: deb_example.labels_and_scales[l] for l in CHOSEN_LABELS},
    "trainset_name": TRAINSET_NAME
}

trials_pspace = hp.pchoice("train_and_test_model", [
    (0.50, {
        "description": "Best: current best model structure with varied dnn, hyperparams and training",
        "model": { 
            "func": make_trained_cnn_model.make_best_model,
            "model_name":               "Model-Search-Variation-On-Best",
            "chosen_features":          CHOSEN_FEATURES,
            "mags_bins":                MAGS_BINS,
            "mags_wrap_phase":          MAGS_WRAP_PHASE,
            "chosen_labels":            CHOSEN_LABELS,
            "trainset_name":            TRAINSET_NAME,
            "cnn_activation":           hp.choice("best_cnn_activation", cnn_activation_choices),
            "dnn_num_layers":           hp.uniformint("best_dnn_num_layers", low=1, high=4),
            "dnn_num_units":            hp.quniform("best_dnn_num_units", low=128, high=512, q=64),
            "dnn_initializer":          hp.choice("best_dnn_initializer", dnn_initializer_choices),
            "dnn_activation":           hp.choice("best_dnn_activation", dnn_activation_choices),
            "dnn_dropout_rate":         hp.quniform("best_dnn_dropout", low=0.3, high=0.7, q=0.1),
            "dnn_num_taper_units":      hp.quniform("best_dnn_taper_units", low=0, high=128, q=32),
            "verbose":                  True
        },
        "optimizer": hp.choice("best_optimizer", [
            {
                "class": optimizers.Adam,
                "learning_rate": hp.choice("best_adam_lr", [
                    hp.qloguniform("best_adam_lr_fixed", **lr_qlogu_kwargs),
                    {
                        "class": optimizers.schedules.ExponentialDecay,
                        "initial_learning_rate": hp.qloguniform("best_adam_exp_lr", **lr_qlogu_kwargs),
                        "decay_steps": 1000,
                        "decay_rate": hp.uniform("best_adam_exp_dr", **lr_exp_decay_rate_kwargs),
                        "staircase": hp.choice("best_adam_exp_staircase", [True, False])
                    },
                    {
                        "class": optimizers.schedules.CosineDecay,
                        "initial_learning_rate": 0.0,
                        "warmup_steps": hp.quniform("best_adam_cos_warmup_steps", **lr_cos_warmup_steps_kwargs),
                        "warmup_target": hp.qloguniform("best_adam_cos_warmup_target", **lr_qlogu_kwargs),
                        "decay_steps": hp.quniform("best_adam_cos_decay_steps", **lr_cos_decay_steps_kwargs),
                        "alpha": hp.choice("best_adam_cos_alpha", lr_cos_alpha_choices),
                    },
                    {
                        "class": optimizers.schedules.PiecewiseConstantDecay,
                        "boundaries": hp.choice("best_adam_pw_boundaries", lr_pw_boundary_choices),
                        "values": hp.choice("best_adam_pw_values", lr_pw_value_choices),                  
                    },
                ])
            },
            {
                "class": optimizers.Nadam,
                "learning_rate": hp.choice("best_nadam_lr", [
                    hp.qloguniform("best_nadam_lr_fixed", **lr_qlogu_kwargs),
                    {
                        "class": optimizers.schedules.ExponentialDecay,
                        "initial_learning_rate": hp.qloguniform("best_nadam_exp_lr", **lr_qlogu_kwargs),
                        "decay_steps": 1000,
                        "decay_rate": hp.uniform("best_nadam_exp_dr", **lr_exp_decay_rate_kwargs),
                        "staircase": hp.choice("best_nadam_exp_staircase", [True, False])
                    },
                    {
                        "class": optimizers.schedules.CosineDecay,
                        "initial_learning_rate": 0.0,
                        "warmup_steps": hp.quniform("best_nadam_cos_warmup_steps", **lr_cos_warmup_steps_kwargs),
                        "warmup_target": hp.qloguniform("best_nadam_cos_warmup_target", **lr_qlogu_kwargs),
                        "decay_steps": hp.quniform("best_nadam_cos_decay_steps", **lr_cos_decay_steps_kwargs),
                        "alpha": hp.choice("best_nadam_cos_alpha", lr_cos_alpha_choices),
                    },
                    {
                        "class": optimizers.schedules.PiecewiseConstantDecay,
                        "boundaries": hp.choice("best_nadam_pw_boundaries", lr_pw_boundary_choices),
                        "values": hp.choice("best_nadam_pw_values", lr_pw_value_choices),
                    },
                ])
            },
            # { # Covers both vanilla SGD and Nesterov momentum
            #     "class": optimizers.SGD,
            #     "learning_rate":            hp.qloguniform("best_sgd_lr", **lr_qlogu_kwargs),
            #     "momentum":                 hp.uniform("best_sgd_momentum", **sgd_momentum_uniform_kwargs),
            #     "nesterov":                 hp.choice("best_sgd_nesterov", [True, False])
            # }
        ]),
        "loss_function":                hp.choice("best_loss", loss_function_choices),
    }),
    (0.50, {
        "description": "Free: explore model structure and hyperparams",
        "model": {
            "func": modelling.build_mags_ext_model,
            "name": "Model-Search-Free-Structure",
            "mags_input": {
                "func": modelling.mags_input_layer,
                "shape": (MAGS_BINS, 1),
            },
            "ext_input": {
                "func": modelling.ext_input_layer,
                "shape": (len(CHOSEN_FEATURES), 1),
            },
            "mags_layers": hp.choice("free_mags_layers", [
                {
                    # Pairs of Conv1ds with fixed filters/kernels/strides and optional pooling layers
                    "func": model_search_helpers.cnn_fixed_pairs_with_pooling,
                    "num_pairs":            hp.uniformint("free_cnn_fixed_num_pairs", low=2, high=4),
                    "filters":              hp.quniform("free_cnn_fixed_filters", low=32, high=96, q=16),
                    "kernel_size":          hp.quniform("free_cnn_fixed_kernel_size", low=4, high=12, q=4),
                    # For strides None defers to strides_fraction
                    "strides":              hp.pchoice("free_cnn_fixed_strides", cnn_strides_pchoices),
                    "strides_fraction":     hp.quniform("free_cnn_fixed_strides_fraction", **cnn_strides_fraction_kwargs),
                    "padding":              hp.choice("free_cnn_fixed_padding", cnn_padding_choices),
                    "activation":           hp.choice("free_cnn_fixed_activation", cnn_activation_choices),
                    "pooling_type":         hp.choice("free_cnn_fixed_pooling_type", cnn_pooling_type_choices),
                    "trailing_pool":        hp.choice("free_cnn_fixed_trailing_pool", [True, False]),
                },
                {
                    # Pairs of Conv1ds with doubling filters & halving kernels/strides per pair
                    # and optional pooling layers
                    "func": model_search_helpers.cnn_scaled_pairs_with_pooling,
                    "num_pairs":            hp.choice("free_cnn_scaled_num_pairs", [3]),
                    "filters":              hp.quniform("free_cnn_scaled_filters", low=16, high=32, q=16),
                    "kernel_size":          hp.quniform("free_cnn_scaled_kernel_size", low=8, high=32, q=8),
                    # For strides None defers to strides_fraction
                    "strides":              hp.pchoice("free_cnn_scaled_strides", cnn_strides_pchoices),
                    "strides_fraction":     hp.quniform("free_cnn_scaled_strides_fraction", **cnn_strides_fraction_kwargs),
                    "scaling_multiplier":   2,
                    "padding":              "same",
                    "activation":           hp.choice("free_cnn_scaled_activation", cnn_activation_choices),
                    "pooling_type":         hp.choice("free_cnn_scaled_pooling_type", cnn_pooling_type_choices),
                    "trailing_pool":        hp.choice("free_cnn_scaled_trailing_pool", [True, False]),
                },
                {
                    # Randomized CNN with/without pooling.
                    "func": model_search_helpers.cnn_with_pooling,
                    "num_layers":           hp.uniformint("free_cnn_num_layers", low=3, high=5),
                    "filters":              hp.quniform("free_cnn_filters", low=32, high=64, q=16),
                    "kernel_size":          hp.quniform("free_cnn_kernel_size", low=4, high=16, q=4),
                    # For strides None defers to strides_fraction
                    "strides":              hp.pchoice("free_cnn_strides", cnn_strides_pchoices),
                    "strides_fraction":     hp.quniform("free_cnn_strides_fraction", **cnn_strides_fraction_kwargs),
                    "padding":              hp.choice("free_cnn_padding", cnn_padding_choices),
                    "activation":           hp.choice("free_cnn_activation", cnn_activation_choices),
                    "pooling_ixs":          hp.choice("free_cnn_pooling_ixs", [None, [2], [2, 5]]),
                    "pooling_type":         hp.choice("free_cnn_pooling_type", cnn_pooling_type_choices),
                },
            ]),
            "ext_layers": None,
            "dnn_layers": hp.choice("dnn_layers", [
                {
                    "func": model_search_helpers.dnn_with_taper,
                    "num_layers":           hp.uniformint("free_dnn_num_layers", low=1, high=4),
                    "units":                hp.quniform("free_dnn_units", low=128, high=512, q=64),
                    "kernel_initializer":   dnn_kernel_initializer_choice,
                    "activation":           hp.choice("free_dnn_activation", dnn_activation_choices),
                    "dropout_rate":         hp.quniform("free_dnn_dropout", low=0.3, high=0.7, q=0.1),
                    "taper_units":          hp.quniform("free_dnn_taper", low=0, high=128, q=32),
                },
            ]),
            "output": {
                "func": modelling.output_layer,
                "metadata":                 metadata,
                "kernel_initializer":       dnn_kernel_initializer_choice,
                "activation":               "linear"
            },
        },
        "optimizer": hp.choice("free_optimizer", [
            {
                "class": optimizers.Adam,
                "learning_rate":            hp.qloguniform("free_adam_lr", **lr_qlogu_kwargs)
            },
            {
                "class": optimizers.Nadam,
                "learning_rate":            hp.qloguniform("free_nadam_lr", **lr_qlogu_kwargs)
            },
            # { # Covers both vanilla SGD and Nesterov momentum
            #     "class": optimizers.SGD,
            #     "learning_rate":            hp.qloguniform("free_sgd_lr", **sgd_lr_qlogu_kwargs),
            #     "momentum":                 hp.uniform("free_sgd_momentum", **sgd_momentum_uniform_kwargs),
            #     "nesterov":                 hp.choice("free_sgd_nesterov", [True, False])
            # }
        ]),
        "loss_function":                hp.choice("free_loss", loss_function_choices), 
    })
])


# -----------------------------------------------------------
# Trials functions
# -----------------------------------------------------------
def train_and_test_model(trial_kwargs):
    """
    Evaluate a single set of hyperparams by building, training and evaluating a model on them.
    """
    print("\n" + "-"*80 + "\n",
          "Evaluating model and hyperparameters based on the following trial_kwargs:\n",
          json.dumps(trial_kwargs, indent=4, sort_keys=False, default=str) + "\n")

    weighted_loss = candidate = None
    status = STATUS_FAIL

    # Reset so shuffling & other tf "random" behaviour is repeated for each trial
    tf.random.set_seed(SEED)

    # Set up the training and validation dataset pipelines
    # Redo this every trial so we can potentially include these pipeline params in the search
    tr_map_func = deb_example.create_map_func(mags_bins=MAGS_BINS, mags_wrap_phase=MAGS_WRAP_PHASE,
                                ext_features=CHOSEN_FEATURES, labels=CHOSEN_LABELS,
                                augmentation_callback=make_trained_cnn_model.augmentation_callback)
    train_ds, train_ct = [tf.data.TFRecordDataset] * 2, [int] * 2
    for ix, (dn, dd) in enumerate(zip(["training", "validation"],[TRAINSET_DIR, VALIDSET_DIR])):
        files = list(dd.glob(TRAINSET_GLOB_TERM))
        (train_ds[ix], train_ct[ix]) = deb_example.create_dataset_pipeline(
                    files, BATCH_FRACTION, tr_map_func, None, True, True, MAX_BUFFER_SIZE,seed=SEED)
        print(f"Found {train_ct[ix]:,} {dn} instances over {len(files)}",
              f"tfrecord file(s) matching glob '{TRAINSET_GLOB_TERM}' within", dd)

    # Set up the training optimizer, loss and metrics
    optimizer = model_search_helpers.get_trial_value(trial_kwargs, "optimizer")
    loss_function = model_search_helpers.get_trial_value(trial_kwargs, "loss_function")
    if not isinstance(loss_function, list):
        loss_function = [loss_function]
    fixed_metrics = ["mae", "mse", "r2_score"]

    try:
        # Build and Compile the trial model
        # always use the same metrics as we use them for trial evaluation
        candidate = model_search_helpers.get_trial_value(trial_kwargs, "model", False)
        candidate.compile(optimizer=optimizer, loss=loss_function, metrics=fixed_metrics)

        # Capture a summary of the newly built model
        with StringIO() as stream:
            candidate.summary(line_length=120, show_trainable=True,
                              print_fn=lambda line: stream.write(line + "\n"))
            model_summary = stream.getvalue()
        print(model_summary, "\n")

        train_callbacks = [
            cb.EarlyStopping("val_loss", min_delta=5e-5, restore_best_weights=True,
                             patience=TRAIN_PATIENCE, start_from_epoch=5, verbose=1),
            TrainingTimeoutCallback(TRAIN_TIMEOUT, verbose=1)
        ]
        candidate.fit(x=train_ds[0], epochs=TRAINING_EPOCHS, callbacks=train_callbacks,
                      validation_data=train_ds[1], verbose=2)

        print(f"\nTrial evaluation of model against {test_ct[0]} test dataset instances.")
        results = candidate.evaluate(x=test_ds[0], y=None, verbose=2)

        print(f"\n'For info' evaluation against {test_ct[1]} formal-test dataset instances.")
        candidate.evaluate(x=test_ds[1], y=None, verbose=2)

        # Out final loss is always MAE from metrics. This allows us to vary the
        # training loss function while using a consistent metric for trial evaluation.
        mae = results[1 + fixed_metrics.index("mae")]
        mse = results[1 + fixed_metrics.index("mse")]

        # The trial can be evaluated on a "weighted loss"; the loss modified with a penalty
        # on model complexity/#params (which is approximated from the number of trainable params).
        weights = int(sum(np.prod(s) for s in [w.shape for w in candidate.trainable_weights]))
        params = np.log(weights)
        weighted_loss = mse * params
        status = STATUS_OK
        features = sum(candidate.get_layer(f"{n}-Input").output.shape[1] for n in ["Mags", "Ext"])
        aic = features*np.log(mse) + 2*params
        bic = features*np.log(mse) + np.log(features)*params
        print(f"""
{'-'*80}
Trial result: MAE = {mae:.6f} and MSE = {mse:.6f}

count(trainable weights) = {weights:,d} yielding params(ln[weights]) = {params:.6f} and:
              weighted loss(MSE*params) = {weighted_loss:6f}
              AIC = {aic:,.3f}
              BIC = {bic:,.3f}
{'-'*80}
""")

    except Exception as exc:
        # Log then rethrow any exception
        print(f"\n*** Training failed! *** Caught a {type(exc).__name__}")
        if isinstance(exc, OpError):
            print(f"Details: {exc.op} / {exc.message}")
        else:
            print(f"Details: {exc}")
        print(f"The call stack is: {traceback.print_exc()}\n")
        raise

    # Make sure anything returned is easily serialized into a pickle so we can save progress.
    return { "loss": mae, "status": status, "mae": mae, "mse": mse,
            "weighted_loss": weighted_loss, "AIC": aic, "BIC": bic,
            "model_summary": model_summary }


def early_stopping_to_report_progress(this_trials, *early_stop_args):
    """
    Callback for early stopping. We don't use it for early stopping but it is
    useful for reporting on the current status of the trials without the
    progress_bar making a mess when capturing the stdout output to a file.
    We also use it to save the current best model details & params as we go.
    """
    no_no_dont_stop = False # Will stop the trial if set to True
    if this_trials and this_trials.best_trial:
        # tids seem to be zero based; +1 for iteration
        best_iter = this_trials.best_trial.get("tid", None) + 1
        br = this_trials.best_trial.get("result", {})
        this_iter = len(this_trials._ids or []) # pylint: disable=protected-access
        print("\n" + "="*80 + "\n",
            f"[{this_iter}/{MAX_HYPEROPT_EVALS}] Best Trial: #{best_iter},",
            f"status={br.get('status', None)}, loss={br.get('loss', 0):.6f},",
            f"MAE={br.get('mae', 0):.6f}, MSE={br.get('mse', 0):.6f}",
            "\n" + "="*80 + "\n")

        if this_iter == best_iter:
            # This is the new best: persist its details so we don't loose them
            model_summary = br.get("model_summary", None)
            summary_file = results_dir / "best_model_summary.txt"
            if model_summary:
                print(f"Saving model summary to {summary_file}")
                with open(summary_file, mode="w", encoding="utf8") as f:
                    f.write(model_summary)
            else:
                summary_file.unlink(missing_ok=True)

            # We have to convert the misc vals dict, which has items like "activation": [2]
            # into the form returned by fmin; the equivalent dict where above is "activation": 2
            # With that we can use space_eval() func to give us the actual values from the vals
            if "misc" in this_trials.best_trial:
                vals = this_trials.best_trial["misc"].get("vals", {})
                vals = { k: v[0] for (k, v) in vals.items() if v }
                if vals:
                    params = space_eval(trials_pspace, vals)
                    params_file = results_dir / "best_model_params.json"
                    print(f"Saving parameters to {params_file}")
                    with open(params_file, mode="w", encoding="utf8") as f:
                        json.dump(params, f, indent=4, default=str)
    return no_no_dont_stop, early_stop_args


# -----------------------------------------------------------
# Conduct the trials
# -----------------------------------------------------------
if __name__ == "__main__":

    trials_log_file = results_dir / "trials.log"
    trials_save_file = results_dir / "trials.pkl"
    if trials_save_file.exists():
        print(f"\nFound an existing trials progress file ({trials_save_file}) from which",
              "hyperopt will attempt to resume the trials.\nIf you don't want to resume",
              "these trials now is your chance to delete this progress file before continuing.")
        input("Press enter to continue or Ctrl+C | Ctrl+Z to exit.")

    resume = trials_save_file.exists()
    with redirect_stdout(Tee(open(trials_log_file, "a" if resume else "w", encoding="utf8"))):
        if resume:
            print("\nResuming the previous Hyperopt trials...\n")
        else:
            print("\nStarting a new Hyperopt trials\n")

        # This appears to be a known false alarm raised from tf 2.16 which can safely be ignored
        # See https://github.com/tensorflow/tensorflow/issues/62963
        filterwarnings("ignore", "Your input ran out of data; interrupting training", UserWarning)

        # The Tree-structured Parzen Estimator (TPE) Hyperparameter Optimization algorithm (HOA)
        # see
        # https://proceedings.neurips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
        # https://proceedings.mlr.press/v28/bergstra13.html
        #
        # Tree-structured refers to the configuration space, whereby some parameters are only
        # well-defined when other parameters have particular values (e.g. #units in DNN layer 3
        # is only well defined with #layers >= 3). In this case #units is a leaf & #layers a node.
        #
        # 1 initial trial steps are a stochastic search over the parameter space
        #   - models are evaluated against the desired objective/loss function
        #	- this initializes the trials history (parameter sets & objective value)
        #   - leads to an approximation of regions of the param space which lead to good/bad models
        # 2 split the parameter space based on some quantile threshold \gamma
        #   - (e.g. \gamma=0.2 will split it 20%/80%)
        #   - Parzen Estimation (aka Kernel Density Estimation [KDE])
        #     - density estimation; an example is a histogram - a means of measuring the density
        #       (# measurements) for each "bin" within a range to estimate their distribution
        #     - estimators based on gaussian KDEs
        #     - fit a Gaussian KDE over each of the good and bad distributions
        #   - l(x) (aka "good" distribution) the ##% of each param values which evaluate the best
        #   - g(x) (aka "bad" distribution) the remaining param values which evaluate less well
        # 3 determine the next potentialy "best" parameter set to test
        #   - for each param x;
        #	  - draw random samples from l(x) (good dist)
        #     - evaluate each sample wrt l(x)/g(x) and select value which maximizes this
        # 4 derive model from selected set
        #	- evaluate model against desired objective/loss function
        #	- result added to trials history
        # 5 repeat steps 2 to 4 until total number of trials reached or params exhausted
        # 6 select parameter set associated with the "best" objective measurement
        #
        # In the hyperopt paper (Bergstra+2013hyperopt) this is summarized as
        # - on each iteration, for each hyperparameter
        #   - fits a Gaussian Mixture Model (GMM) - l(x) - to the set of hyperparam values with
        #     the smallest loss values ("good" dist)
        #   - fits a GMM - g(x) - to the remaining hyperparam values ("bad" dist)
        #   - chooses the hyperparam value x which maximizes l(x)/g(x)
        best = fmin(fn = train_and_test_model,
                    space = trials_pspace,
                    trials = None, # Have fmin create new or deserialize the saved trials
                    algo = tpe.suggest,
                    max_evals = MAX_HYPEROPT_EVALS,
                    loss_threshold = HYPEROPT_LOSS_TH,
                    catch_eval_exceptions = True,
                    rstate=np.random.default_rng(SEED),
                    early_stop_fn=early_stopping_to_report_progress,
                    trials_save_file=f"{trials_save_file}",
                    verbose=True,
                    show_progressbar=False) # Can't use progressbar as it makes a mess of logging

        # Report on the outcome (we saved the best to files as we went along)
        best_params = space_eval(trials_pspace, best)
        print("\nBest model hyperparameter set is:\n"
            + json.dumps(best_params, indent=4, sort_keys=False, default=str))
