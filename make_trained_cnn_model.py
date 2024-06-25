"""
Trains a regression CNN to estimate fitting parameters from folded dEB light curves
"""
#pylint: disable=line-too-long
from pathlib import Path
import os
import random as python_random
import json
from datetime import datetime, timezone
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorboard
import keras
from keras import layers, optimizers, callbacks

from ebop_maven import modelling, deb_example, plotting
from ebop_maven.libs.tee import Tee
from ebop_maven.libs.keras_custom.metrics import MeanAbsoluteErrorForLabel
import model_testing

# Configure the inputs and outputs of the model
CHOSEN_FEATURES = []
MAGS_BINS = 4096
MAGS_WRAP_PHASE = 0.75
CHOSEN_LABELS = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "bP"]
TRAINSET_SIZE = "250k"

MODEL_NAME = f"CNN-New-Ext{len(CHOSEN_FEATURES)}-{'-'.join(CHOSEN_LABELS[5:])}-" \
                            + f"{MAGS_BINS}-{MAGS_WRAP_PHASE}-{TRAINSET_SIZE}"
MODEL_FILE_NAME = "default-model"
SAVE_DIR = Path("./drop/training") / MODEL_NAME.lower()
PLOTS_DIR = SAVE_DIR / "plots"

# We can now specify paths to train/val/test datasets separately for greater flexibility.
TRAINSET_NAME = "formal-training-dataset-" + TRAINSET_SIZE
TRAINSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "training"
VALIDSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "validation"
TESTSET_DIR = Path(".") / "datasets" / "synthetic-mist-tess-dataset"

TRAINING_EPOCHS = 250           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
EARLY_STOPPING_PATIENCE = 15    # Number of epochs w/o improvement before stopping
ENFORCE_REPEATABILITY = True    # If true, avoid GPU/CUDA cores for repeatable results
SEED = 42                       # Standard random seed ensures repeatable randomization
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

OPTIMIZER = optimizers.Nadam(learning_rate=5e-4)
LOSS = ["mae"]
METRICS = ["mse"] #+ [MeanAbsoluteErrorForLabel(CHOSEN_LABELS.index(l), l) for l in CHOSEN_LABELS]

# This gives the option of tweaking the emphasis across the labels when training/reducing the loss
CLASS_WEIGHTS = { CHOSEN_LABELS.index(l): 1 for l in CHOSEN_LABELS } # Currently all the same

# ReLU is widely used default for CNN/DNNs.
# Otherwise, may need to specify each layer separately as dims different.
# LeakyReLU addresses issue of dead neurons & PReLU similar but trains alpha param
CNN_PADDING = "same"
CNN_ACTIVATE = "relu"

# For the dense layers: "glorot_uniform" (def) "he_normal", "he_uniform" (he_ goes well with ReLU)
DNN_INITIALIZER = "he_uniform"
DNN_ACTIVATE = "leaky_relu"
DNN_NUM_UNITS = 256
DNN_NUM_FULL_LAYERS = 2
DNN_DROPOUT_RATE = 0.5
DNN_NUM_TAPER_UNITS = 64

def make_best_model(chosen_features: list[str]=CHOSEN_FEATURES,
                    mags_bins: int=MAGS_BINS,
                    mags_wrap_phase: float=MAGS_WRAP_PHASE,
                    chosen_labels: list[str]=CHOSEN_LABELS,
                    trainset_name: str=TRAINSET_NAME,
                    cnn_padding: str=CNN_PADDING,
                    cnn_activation: str=CNN_ACTIVATE,
                    dnn_num_layers: int=DNN_NUM_FULL_LAYERS,
                    dnn_num_units: int=DNN_NUM_UNITS,
                    dnn_initializer: str=DNN_INITIALIZER,
                    dnn_activation: str=DNN_ACTIVATE,
                    dnn_dropout_rate: float=DNN_DROPOUT_RATE,
                    dnn_num_taper_units: int=DNN_NUM_TAPER_UNITS,
                    model_name: str=MODEL_NAME,
                    verbose: bool=False):
    """
    Helper function for building the current best performing model. 
    Publish model from a function, rather than inline, so it can be shared with model_search.
    """
    # pylint: disable=too-many-arguments, too-many-locals, dangerous-default-value
    print("\nBuilding the best known CNN model for predicting:", ", ".join(chosen_labels))
    metadata = { # This will augment the model, giving an Estimator context information
        "extra_features_and_defaults": 
                    {f: deb_example.extra_features_and_defaults[f] for f in chosen_features },
        "mags_bins": mags_bins,
        "mags_wrap_phase": mags_wrap_phase,
        "labels_and_scales": {l: deb_example.labels_and_scales[l] for l in chosen_labels},
        "trainset_name": trainset_name,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    best_model = modelling.build_mags_ext_model(
        mags_input=modelling.mags_input_layer(shape=(mags_bins, 1), verbose=verbose),
        ext_input=modelling.ext_input_layer(shape=(len(chosen_features), 1), verbose=verbose),
        mags_layers=[
            modelling.conv1d_layers(2, 16, 32, 2, cnn_padding, cnn_activation, "Conv-1-", verbose),
            modelling.pooling_layer(layers.MaxPool1D, 2, 2, "Pool-1", verbose),
            modelling.conv1d_layers(2, 32, 16, 2, cnn_padding, cnn_activation, "Conv-2-", verbose),
            modelling.pooling_layer(layers.MaxPool1D, 2, 2, "Pool-2", verbose),
            modelling.conv1d_layers(2, 64, 8, 2, cnn_padding, cnn_activation, "Conv-3-", verbose),
            modelling.pooling_layer(layers.MaxPool1D, 2, 2, "Pool-3", verbose),
            modelling.conv1d_layers(2, 128, 4, 2, cnn_padding, cnn_activation, "Conv-4-", verbose),
        ],
        dnn_layers=[
            modelling.hidden_layers(int(dnn_num_layers), int(dnn_num_units),
                                    dnn_initializer, dnn_activation,
                                    dnn_dropout_rate, ("Hidden-", "Dropout-"), verbose),
            # "Buffer" between the DNN+Dropout and the output layer; this non-dropout NN layer
            # consistently gives a small, but significant improvement to the trained loss.
            modelling.hidden_layers(1, int(dnn_num_taper_units), dnn_initializer, dnn_activation,
                                    0, ("Taper-",), verbose) if dnn_num_taper_units else None
        ],
        output=modelling.output_layer(metadata, dnn_initializer, "linear", "Output", verbose),
        post_build_step=None,
        name=model_name,
        verbose=verbose
    )
    if verbose:
        print(f"Have built the model {best_model.name}\n")
    return best_model


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(Tee(open(SAVE_DIR / "make_trained_cnn_model.log", "w", encoding="utf8"))):
        print("\n".join(f"{lib.__name__} v{lib.__version__}" for lib in [tf, tensorboard, keras]))
        if ENFORCE_REPEATABILITY:
            # Extreme, but it stops TensorFlow/Keras from using (even seeing) the GPU.
            # Slows training down massively (by 3-4 times) but should avoid GPU memory
            # constraints! Necessary if repeatable results are required (Keras advises
            # that out of order processing within GPU/CUDA can lead to varying results).
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print(f"Found {len(tf.config.list_physical_devices('GPU'))} GPU(s)\n")

        # -----------------------------------------------------------
        # Set up the training and validation dataset pipelines
        # -----------------------------------------------------------
        print("Picking up training and validation datasets.")
        datasets = [tf.data.TFRecordDataset] * 2
        counts = [int] * 2
        ROLL_MAX = int(9 * (MAGS_BINS/1024))
        map_func = deb_example.create_map_func(mags_bins=MAGS_BINS,
                                            mags_wrap_phase=MAGS_WRAP_PHASE,
                                            ext_features=CHOSEN_FEATURES,
                                            labels=CHOSEN_LABELS,
                                            noise_stddev=lambda: 0.005,
                                            roll_steps=lambda: tf.random.uniform(
                                                            [], -ROLL_MAX, ROLL_MAX+1, tf.int32))
        for ds_ix, (label, set_dir) in enumerate([("training", TRAINSET_DIR),
                                                ("valiation", VALIDSET_DIR)]):
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
            print(f"Found {counts[ds_ix]:,} {label} insts over {len(files)} tfrecords in", set_dir)

        # -----------------------------------------------------------
        # Define the model
        # -----------------------------------------------------------
        model = make_best_model(verbose=True)
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
                                    patience=EARLY_STOPPING_PATIENCE, verbose=1),
            callbacks.CSVLogger(SAVE_DIR / "training-log.csv")
        ]

        print(f"\nTraining the model on {counts[0]} training and {counts[1]} validation instances.")
        try:
            # You may see the following warning while training, which can safely be ignored;
            #   UserWarning: Your input ran out of data; interrupting training
            history = model.fit(x = datasets[0],    # pylint: disable=invalid-name
                                epochs = TRAINING_EPOCHS,
                                callbacks = CALLBACKS,
                                class_weight=CLASS_WEIGHTS,
                                validation_data = datasets[1],
                                verbose=2)

            # Plot the learning curves
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(history.history['loss'], label="training")
            ax.plot(history.history['val_loss'], label="validation")
            plotting.format_axes(ax, xlabel="Epoch", ylabel="Loss", legend_loc="best")
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(PLOTS_DIR / f"{MODEL_FILE_NAME}-learning-curves.eps", dpi=300)
        except tf.errors.InvalidArgumentError as exc:
            if ("lc" in exc.message or "mags" in exc.message) \
                    and "Can't parse serialized" in exc.message:
                msg = exc.message + "\n*** Probably an incompatible serialized mags feature length."
                raise tf.errors.InvalidArgumentError(exc.node_def, exc.op, msg, exc.args) from exc

        # Save the newly trained model
        model_file = SAVE_DIR / f"{MODEL_FILE_NAME}.keras"
        modelling.save_model(model_file, model)
        print(f"\nSaved model '{MODEL_NAME}' to: {model_file}")

        # -----------------------------------------------------------
        # Test the newly saved model against various test datasets
        # -----------------------------------------------------------
        # We use scaled prediction so the MAE/MSE is comperable with model.fit() & model.evaluate()
        # Test against the synthetic test dataset
        print(f"\n *** Running tests against {TESTSET_DIR.name}\n")
        model_testing.evaluate_model_against_dataset(model_file, 1, None, TESTSET_DIR, scaled=True)

        # Test against the formal test set of real systems
        with open("./config/formal-test-dataset.json", mode="r", encoding="utf8") as tf:
            targs_config = json.load(tf)
        usable_targs = np.array([t for t, c in targs_config.items() if not c.get("exclude", False)])
        print("\n *** Running tests against formal-test-dataset with no MC-Dropout\n")
        model_testing.evaluate_model_against_dataset(model_file, 1, usable_targs, scaled=True)
        print("\n *** Running tests against formal-test-dataset with 1000 MC-Dropout iterations\n")
        model_testing.evaluate_model_against_dataset(model_file, 1000, usable_targs, scaled=True)
