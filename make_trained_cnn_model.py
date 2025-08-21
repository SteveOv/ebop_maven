"""
Trains a regression CNN to estimate fitting parameters from folded dEB light curves
"""
#pylint: disable=line-too-long
from pathlib import Path
import sys
import json
from inspect import getsource
from datetime import datetime, timezone
from contextlib import redirect_stdout
from warnings import filterwarnings

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorboard
import keras
from keras import layers, optimizers, callbacks
from keras.src.layers.pooling.base_pooling import BasePooling

from ebop_maven import modelling, deb_example
from traininglib.keras_custom.metrics import MeanAbsoluteErrorForLabel
from traininglib import plotting
from traininglib.datasets import create_map_func, create_dataset_pipeline
from traininglib.tee import Tee
import model_testing

# Configure the inputs and outputs of the model
CHOSEN_FEATURES = []
MAGS_BINS = 4096
MAGS_WRAP_PHASE = None # None indicates wrap to centre on midpoint between eclipses
CHOSEN_LABELS = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "bP"]
OUTPUT_ACTIVATIONS = ["softplus"]*3 + ["linear"]*3
TRAINSET_SUFFIX = "500k"

MODEL_NAME = f"CNN-New-Ext{len(CHOSEN_FEATURES)}-{'-'.join(CHOSEN_LABELS[5:])}-" \
                            + f"{MAGS_BINS}-{MAGS_WRAP_PHASE}-{TRAINSET_SUFFIX}"
MODEL_FILE_NAME = "default-model"
SAVE_DIR = Path("./drop/training") / MODEL_NAME.lower()
PLOTS_DIR = SAVE_DIR / "plots"

# We can now specify paths to train/val/test datasets separately for greater flexibility.
TRAINSET_NAME = "formal-training-dataset-" + TRAINSET_SUFFIX
TRAINSET_GLOB_TERM = "trainset*.tfrecord"
TRAINSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "training"
TRAINSET_PIPELINE_AUGS = True
VALIDSET_DIR = Path(".") / "datasets" / TRAINSET_NAME / "validation"
VALIDSET_PIPELINE_AUGS = True
TESTSET_DIR = Path(".") / "datasets" / "synthetic-mist-tess-dataset"

TRAINING_EPOCHS = 250           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
ES_PATIENCE = 5                 # Number of epochs w/o val_loss improvement before stopping
ES_MIN_DELTA = 0.0001           # Minimum val_loss delta to be considered an improvment
SEED = 42                       # Standard random seed ensures repeatable randomization

# Sets the random seed on python, numpy and keras's backend library (in this case tensorflow)
keras.utils.set_random_seed(SEED)

# Make GPU ops as deterministic as possible, for repeatable results, at the expense of performance.
# Note that MaxPool1D layers are incompatible with this setting; they cause the following error
# "GPU MaxPool gradient ops do not yet have a deterministic XLA implementation", however AvgPool1D
# are fine. Even with this setting, GPUs are seen only if CUDA_VISIBLE_DEVICES isn't set to -1.
#tf.config.experimental.enable_op_determinism()

# This schedule is effectively init_rate * 0.94^epoch (so is reduced by ~10 in 37 epochs)
LR = optimizers.schedules.ExponentialDecay(1e-3, decay_steps=1000, decay_rate=0.94)
OPTIMIZER = optimizers.Nadam(learning_rate=LR)
LOSS = ["mae"]
METRICS = ["mse"] #+ [MeanAbsoluteErrorForLabel(CHOSEN_LABELS.index(l), l) for l in CHOSEN_LABELS]

# This gives the option of tweaking the emphasis across the labels when training/reducing the loss
CLASS_WEIGHTS = { CHOSEN_LABELS.index(l): 1 for l in CHOSEN_LABELS } # Currently all the same

# ReLU is widely used default for CNN/DNNs.
# Otherwise, may need to specify each layer separately as dims different.
# LeakyReLU addresses issue of dead neurons & PReLU similar but trains alpha param
CNN_PADDING = "same"
CNN_ACTIVATE = "relu"
CNN_POOLING_TYPE = layers.MaxPool1D # pylint: disable=invalid-name

# For the dense layers: "glorot_uniform" (def) "he_normal", "he_uniform" (he_ goes well with ReLU)
DNN_INITIALIZER = "he_uniform"
DNN_ACTIVATE = "leaky_relu"
DNN_NUM_UNITS = 256
DNN_NUM_FULL_LAYERS = 2
DNN_DROPOUT_RATE = 0.5
DNN_NUM_TAPER_UNITS = 64

# Control dataset pipeline augmentations applied to each mags_feature
NOISE_MAX = 0.030               # max random sigma value (in mag) of Gaussian noise to add
ROLL_SIGMA = 0.066              # 1-sigma of random roll (in phase) of feature left or right
YSHIFT_SIGMA = 0.030            # 1-sigma of random y-shift (in mag) of feature up or down
@tf.function
def augmentation_callback(mags_feature: tf.Tensor) -> tf.Tensor:
    """
    Dataset pipeline augmentation function which is called from the map_func. Updates the
    mags_feature with random amounts of additive Gaussian noise, a random roll left or right
    and a random "vertical" magnitude shift up or down.
    """
    noise_stddev = tf.random.uniform([], 0.001, NOISE_MAX, tf.float32)
    if noise_stddev != 0:
        mags_feature += tf.random.normal(mags_feature.shape, stddev=noise_stddev)
    roll_by = int(MAGS_BINS * tf.random.normal([], stddev=ROLL_SIGMA))
    if roll_by != 0:
        mags_feature = tf.roll(mags_feature, [roll_by], axis=[0])
    y_shift = tf.random.normal([], stddev=YSHIFT_SIGMA)
    if y_shift != 0:
        mags_feature += y_shift
    return mags_feature


def make_best_model(chosen_features: list[str]=CHOSEN_FEATURES,
                    mags_bins: int=MAGS_BINS,
                    mags_wrap_phase: float=MAGS_WRAP_PHASE,
                    chosen_labels: list[str]=CHOSEN_LABELS,
                    trainset_name: str=TRAINSET_NAME,
                    cnn_padding: str=CNN_PADDING,
                    cnn_activation: str=CNN_ACTIVATE,
                    cnn_pooling: BasePooling=CNN_POOLING_TYPE,
                    dnn_num_layers: int=DNN_NUM_FULL_LAYERS,
                    dnn_num_units: int=DNN_NUM_UNITS,
                    dnn_initializer: str=DNN_INITIALIZER,
                    dnn_activation: str=DNN_ACTIVATE,
                    dnn_dropout_rate: float=DNN_DROPOUT_RATE,
                    dnn_num_taper_units: int=DNN_NUM_TAPER_UNITS,
                    model_name: str=MODEL_NAME,
                    output_activations: list[str]=OUTPUT_ACTIVATIONS,
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
        "created_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    best_model = modelling.build_mags_ext_model(
        mags_input=modelling.mags_input_layer(shape=(mags_bins, 1), verbose=verbose),
        ext_input=modelling.ext_input_layer(shape=(len(chosen_features), 1), verbose=verbose),
        mags_layers=[
            modelling.conv1d_layers(2, 16, 32, 2, cnn_padding, cnn_activation, "Conv-1-", verbose),
            modelling.pooling_layer(cnn_pooling, 2, 2, cnn_padding, "Pool-1", verbose),
            modelling.conv1d_layers(2, 32, 16, 2, cnn_padding, cnn_activation, "Conv-2-", verbose),
            modelling.pooling_layer(cnn_pooling, 2, 2, cnn_padding, "Pool-2", verbose),
            modelling.conv1d_layers(2, 64, 8, 2, cnn_padding, cnn_activation, "Conv-3-", verbose),
            modelling.pooling_layer(cnn_pooling, 2, 2, cnn_padding, "Pool-3", verbose),
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
        output=modelling.output_layer(metadata, dnn_initializer, output_activations, verbose),
        post_build_step=None,
        name=model_name,
        verbose=verbose
    )
    if verbose:
        print(f"Have built the model {best_model.name}\n")
    return best_model


if __name__ == "__main__":
    print()
    log_file = SAVE_DIR / "make_trained_cnn_model.log"
    if log_file.exists():
        resp = input(f"Training output for model exists in {SAVE_DIR}. Continue & overwrite y/N? ")
        if resp.strip().lower() not in ["y", "yes"]:
            sys.exit()

    print(f"Trained model and logs will be saved to {SAVE_DIR}\n")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(Tee(open(log_file, "w", encoding="utf8"))):
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}\n")

        # CUDA_VISIBLE_DEVICES is set for the ebop_maven conda env. A value of "-1" suppresses GPUs,
        # useful for repeatable results (Keras advises that out of order processing within GPU/CUDA
        # can lead to varying results) and also avoiding memory constraints on smaller GPUs (mine!).
        print("Runtime environment:", sys.prefix.replace("'", ""))
        print("\n".join(f"{lib.__name__} v{lib.__version__}" for lib in [tf, tensorboard, keras]))
        print(f"tensorflow can see {len(tf.config.list_physical_devices('GPU'))} physical GPU(s)")

        # -----------------------------------------------------------
        # Define the model
        # -----------------------------------------------------------
        model = make_best_model(verbose=True)
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
        model.summary()
        try:
            # Can only get this working with specific pydot (1.4) & graphviz (8.0) conda packages.
            # With pip I can't get graphviz beyond 0.2.0 which leads to pydot errors here.
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            keras.utils.plot_model(model, to_file=PLOTS_DIR / f"{MODEL_FILE_NAME}.pdf",
                    show_layer_names=True, show_shapes=True, show_layer_activations=True,
                    show_dtype=False, show_trainable=False, rankdir="TB", dpi=300)
        except ImportError:
            print("Unable to plot_model() without pydot and/or graphviz.")

        # -----------------------------------------------------------
        # Set up the training and validation dataset pipelines
        # -----------------------------------------------------------
        print("\nCreating training and validation dataset pipelines.")
        if MAGS_WRAP_PHASE is None:
            print("The mags features will be centred on the midpoint between eclipses.")
        else:
            print(f"The mags features will be wrapped beyond phase {MAGS_WRAP_PHASE}.")
        datasets, counts = [tf.data.TFRecordDataset] * 2, [int] * 2
        for ix, (label, set_dir, requires_augs) in enumerate([
            ("training", TRAINSET_DIR, TRAINSET_PIPELINE_AUGS),
            ("validation", VALIDSET_DIR, VALIDSET_PIPELINE_AUGS)
        ]):
            files = sorted(set_dir.glob(TRAINSET_GLOB_TERM))
            aug_func = augmentation_callback if requires_augs else None
            if aug_func is not None:
                print(f"\nThe {label} set pipeline will apply augmentations based on",
                      f"NOISE_MAX_SIGMA={NOISE_MAX}, ROLL_SIGMA={ROLL_SIGMA},",
                      f"YSHIFT_SIGMA={YSHIFT_SIGMA} with:\n{getsource(aug_func)}")
            map_func = create_map_func(mags_bins=MAGS_BINS, mags_wrap_phase=MAGS_WRAP_PHASE,
                                       ext_features=CHOSEN_FEATURES, labels=CHOSEN_LABELS,
                                       augmentation_callback=aug_func)
            if ix == 0:
                datasets[ix], counts[ix] = create_dataset_pipeline(files, BATCH_FRACTION, map_func,
                                                                   shuffle=True,
                                                                   reshuffle_each_iteration=True,
                                                                   max_buffer_size=MAX_BUFFER_SIZE,
                                                                   seed=SEED)
            else:
                datasets[ix], counts[ix] = create_dataset_pipeline(files, BATCH_FRACTION, map_func)
            print(f"Found {counts[ix]:,} {label} insts over {len(files)}",
                  f"tfrecords matching glob '{TRAINSET_GLOB_TERM}' within", set_dir)

        # -----------------------------------------------------------
        # Train the model
        # -----------------------------------------------------------
        print("\nTraining:",
              f"epochs={TRAINING_EPOCHS}, patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA}")
        print(f"Optimizer: {OPTIMIZER.name} where LR is",
              LR if isinstance(LR, (int, float)) else f"{LR.name}({vars(LR)})")
        print(f"Loss function {LOSS} and metrics are {METRICS}")

        CALLBACKS = [
            # To use tensorboard make sure the containing conda env is active then run
            # $ tensorboard --port 6006 --logdir ./logs
            # Then start a browser and head to http://localhost:6006
            #callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True),
            callbacks.EarlyStopping("val_loss", restore_best_weights=True, min_delta=ES_MIN_DELTA,
                                    start_from_epoch=50, patience=ES_PATIENCE, verbose=1),
            callbacks.CSVLogger(SAVE_DIR / "training-log.csv")
        ]

        # This appears to be a known false alarm raised from tf 2.16 which can safely be ignored
        # See https://github.com/tensorflow/tensorflow/issues/62963
        filterwarnings("ignore", "Your input ran out of data; interrupting training", UserWarning)

        print(f"\nTraining the model on {counts[0]} training and {counts[1]} validation instances.")
        try:
            history = model.fit(x = datasets[0],    # pylint: disable=invalid-name
                                epochs = TRAINING_EPOCHS,
                                callbacks = CALLBACKS,
                                class_weight=CLASS_WEIGHTS,
                                validation_data = datasets[1],
                                verbose=2)

            # Plot the learning curves
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.plot(history.history['loss'], label="training")
            ax.plot(history.history['val_loss'], label="validation")
            plotting.format_axes(ax, xlabel="Epoch", ylabel="Loss", legend_loc="best")
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(PLOTS_DIR / f"{MODEL_FILE_NAME}-learning-curves.pdf", dpi=300)
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

        print(f"Completed at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}\n")
