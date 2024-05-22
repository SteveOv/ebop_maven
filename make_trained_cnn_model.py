"""
Trains a regression CNN to estimate fitting parameters from folded dEB light curves
"""
#pylint: disable=line-too-long
from pathlib import Path
import os
import random as python_random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorboard
import keras
from keras import layers, optimizers, callbacks

from ebop_maven import modelling, deb_example
import model_testing

# Configure the inputs and outputs of the model
CHOSEN_FEATURES = []
MAGS_BINS = 4096
MAGS_WRAP_PHASE = 0.75
CHOSEN_LABELS = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "bP"]

MODEL_NAME = f"CNN-New-Ext{len(CHOSEN_FEATURES)}-{'-'.join(CHOSEN_LABELS[5:])}-{MAGS_BINS}-{MAGS_WRAP_PHASE}"
MODEL_FILE_NAME = MODEL_NAME.lower()
SAVE_DIR = Path(".") / "drop/"
PLOTS_DIR = SAVE_DIR / "plots"

# We can now specify paths to train/val/test datasets separately for greater flexibility.
TRAINSET_NAME = "formal-training-dataset"
TRAINSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "training"
VALIDSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "validation"
TESTSET_DIR = Path(".") / "datasets" / "synthetic-mist-tess-dataset"

TRAINING_EPOCHS = 100           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
EARLY_STOPPING_PATIENCE = 7     # Number of epochs w/o improvement before stopping
ENFORCE_REPEATABILITY = True    # If true, avoid GPU/CUDA cores for repeatable results
SEED = 42                       # Standard random seed ensures repeatable randomization
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

OPTIMIZER = optimizers.Nadam(learning_rate=5e-4)
LOSS = ["mae"]
METRICS = ["mse"]

# ReLU is widely used default for CNN/DNNs.
# Otherwise, may need to specify each layer separately as dims different.
# LeakyReLU addresses issue of dead neurons & PReLU similar but trains alpha param
CNN_PADDING = "same"
CNN_ACTIVATE = "relu"
DNN_ACTIVATE = "leaky_relu"

# For the dense layers: "glorot_uniform" (def) "he_normal", "he_uniform" (he_ goes well with ReLU)
DNN_INITIALIZER = "he_uniform"
DNN_NUM_FULL_LAYERS = 2
DNN_DROPOUT_RATE=0.5

def make_best_model(chosen_features: list[str],
                    mags_bins: int,
                    mags_wrap_phase: float,
                    chosen_labels: list[str],
                    trainset_name: str,
                    verbose: bool=False):
    """
    Helper function for building the current best performing model. 
    Publish model from a function, rather than inline, so it can be shared with model_search.
    """
    print("\nBuilding the best known CNN model for predicting:", ", ".join(chosen_labels))
    metadata = { # This will augment the model, giving an Estimator context information
        "extra_features_and_defaults": 
                    {f: deb_example.extra_features_and_defaults[f] for f in chosen_features },
        "mags_bins": mags_bins,
        "mags_wrap_phase": mags_wrap_phase,
        "labels_and_scales": {l: deb_example.labels_and_scales[l] for l in chosen_labels},
        "trainset_name": trainset_name
    }
    best_model = modelling.build_mags_ext_model(
        name=MODEL_NAME,
        mags_input=modelling.mags_input_layer(shape=(mags_bins, 1), verbose=verbose),
        ext_input=modelling.ext_input_layer(shape=(len(chosen_features), 1), verbose=verbose),
        mags_layers=[
            modelling.conv1d_layers(2, 16, 32, 2, CNN_PADDING, CNN_ACTIVATE, "Conv-1-", verbose),
            modelling.pooling_layer(layers.MaxPool1D, 2, 2, "Pool-1", verbose),
            modelling.conv1d_layers(2, 32, 16, 2, CNN_PADDING, CNN_ACTIVATE, "Conv-2-", verbose),
            modelling.pooling_layer(layers.MaxPool1D, 2, 2, "Pool-2", verbose),
            modelling.conv1d_layers(2, 64, 8, 2, CNN_PADDING, CNN_ACTIVATE, "Conv-3-", verbose),
            modelling.pooling_layer(layers.MaxPool1D, 2, 2, "Pool-3", verbose),
            modelling.conv1d_layers(2, 128, 4, 2, CNN_PADDING, CNN_ACTIVATE, "Conv-4-", verbose),
        ],
        dnn_layers=[
            modelling.hidden_layers(DNN_NUM_FULL_LAYERS, 256, DNN_INITIALIZER, DNN_ACTIVATE,
                                    DNN_DROPOUT_RATE, ("Hidden-", "Dropout-"), verbose),
            # "Buffer" between the DNN+Dropout and the output layer; this non-dropout NN layer
            # consistently gives a small, but significant improvement to the trained loss.
            modelling.hidden_layers(1, 64, DNN_INITIALIZER, DNN_ACTIVATE, 0, ("Taper-",), verbose)
        ],
        output=modelling.output_layer(metadata, DNN_INITIALIZER, "linear", "Output", verbose),
        verbose=verbose
    )
    if verbose:
        print(f"Have built the model {best_model.name}\n")
    return best_model


if __name__ == "__main__":
    print("\n".join(f"{lib.__name__} v{lib.__version__}" for lib in [tf, tensorboard, keras]))
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
    print("Picking up training/validation/test datasets.")
    datasets = [tf.data.TFRecordDataset] * 3
    counts = [int] * 3
    ROLL_MAX = int(9 * (MAGS_BINS/1024))
    map_func = deb_example.create_map_func(mags_bins=MAGS_BINS,
                                           mags_wrap_phase=MAGS_WRAP_PHASE,
                                           ext_features=CHOSEN_FEATURES,
                                           labels=CHOSEN_LABELS,
                                           noise_stddev=lambda: 0.005,
                                        roll_steps=lambda: tf.random.uniform([], -ROLL_MAX,
                                                                             ROLL_MAX+1, tf.int32))
    for ds_ix, (label, set_dir) in enumerate([("training", TRAINSET_DIR),
                                            ("valiation", VALIDSET_DIR),
                                            ("testing", TESTSET_DIR)]):
        files = list(set_dir.glob("**/*.tfrecord"))
        if ds_ix == 0:
            (datasets[ds_ix], counts[ds_ix]) = \
                deb_example.create_dataset_pipeline(files, BATCH_FRACTION, map_func,
                                                    shuffle=True, reshuffle_each_iteration=True,
                                                    max_buffer_size=MAX_BUFFER_SIZE,
                                                    prefetch=1, seed=SEED)
        else:
            (datasets[ds_ix], counts[ds_ix]) = \
                deb_example.create_dataset_pipeline(files, BATCH_FRACTION, map_func)
        print(f"Found {counts[ds_ix]:,} {label} insts over {len(files)} tfrecord files in", set_dir)

    # -----------------------------------------------------------
    # Define the model
    # -----------------------------------------------------------
    model = make_best_model(CHOSEN_FEATURES, MAGS_BINS, MAGS_WRAP_PHASE, CHOSEN_LABELS,
                            TRAINSET_NAME, verbose=True)
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.summary()

    try:
        # Can only get this working specific pydot (1.4) & graphviz (8.0) conda packages.
        # With pip I can't get graphviz beyond 0.2.0 which leads to pydot errors here.
        # At least with the try block I can degrade gracefully.
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        keras.utils.plot_model(model, to_file=PLOTS_DIR / f"{MODEL_FILE_NAME}.png",
                show_layer_names=True, show_shapes=True, show_layer_activations=True,
                show_dtype=False, show_trainable=False, rankdir="TB", dpi=300)
    except ImportError:
        print("Unable to plot_model() without pydot and/or graphviz.")


    # -----------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------
    CALLBACKS = [
        # To use tensorboard make sure the containing conda env is active then run
        # $ tensorboard --port 6006 --logdir ./logs
        # Then start a browser and head to http://localhost:6006
        #callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True),
        callbacks.EarlyStopping("val_loss", restore_best_weights=True,
                                patience=EARLY_STOPPING_PATIENCE, verbose=1)
    ]

    print(f"\nTraining the model on {counts[0]} training and {counts[1]} validation",
        f"instances, with a further {counts[2]} instances held back for test.")
    try:
        # You may see the following warning while training, which can safely be ignored;
        #   UserWarning: Your input ran out of data; interrupting training
        history = model.fit(x = datasets[0],    # pylint: disable=invalid-name
                            epochs = TRAINING_EPOCHS,
                            callbacks = CALLBACKS,
                            validation_data = datasets[1])
        # Plot the learning curves
        ax = pd.DataFrame(history.history).plot(figsize=(6, 4), xlabel="Epoch", ylabel="Loss")
        ax.get_figure().tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / f"{MODEL_FILE_NAME}-learning-curves.eps", dpi=300)
    except tf.errors.InvalidArgumentError as exc:
        if ("lc" in exc.message or "mags" in exc.message) \
                and "Can't parse serialized" in exc.message:
            msg = exc.message + "\n*** Probable cause: incompatible serialized mags feature length."
            raise tf.errors.InvalidArgumentError(exc.node_def, exc.op, msg, exc.args) from exc

    print(f"\nEvaluating the model on {counts[2]} test instances.")
    model.evaluate(datasets[2], verbose=1)

    # Evaluate against the test dataset filtered to various subsets
    if "bP" in CHOSEN_LABELS:
        ix_bp = CHOSEN_LABELS.index("bP")
        ix_k = CHOSEN_LABELS.index("k")
        files = list(TESTSET_DIR.glob("**/*.tfrecord"))
        for msg, filter_func in [
                    ("Transiting systems",     lambda _, lab: lab[ix_bp] < (1-lab[ix_k])),
                    ("Non-transiting systems", lambda _, lab: lab[ix_bp] >= (1-lab[ix_k]))
            ]:
            print("\nEvaluatiing model on the following subset of the test dataset;", msg)
            (ds_filtered, _) = deb_example.create_dataset_pipeline(files, BATCH_FRACTION,
                                                                   map_func, filter_func)
            model.evaluate(ds_filtered, verbose=1)

    # Save the newly trained model
    model_save_file = SAVE_DIR / f"{MODEL_FILE_NAME}.keras"
    modelling.save_model(model_save_file, model)
    print(f"\nSaved model '{MODEL_NAME}' to: {model_save_file}")

    # -----------------------------------------------------------
    # Tests the newly saved model, within an Estimator, against a dataset of real systems.
    # -----------------------------------------------------------
    # We use scaled prediction so the MAE/MSE is comperable with model.fit() and model.evaluate()
    print("\n *** Running formal test against real data ***")
    model_testing.test_model_against_formal_test_dataset(model_save_file, 1, scaled=True)
    model_testing.test_model_against_formal_test_dataset(model_save_file, 1000, scaled=True)
