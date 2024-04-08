"""
Trains a regression CNN to estimate fitting parameters from folded dEB light curves
"""
from pathlib import Path
import os
import random as python_random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorboard
import keras
from keras import layers, initializers, optimizers, callbacks, metrics
from keras.utils import plot_model

from ebop_maven import modelling
from ebop_maven.libs import deb_example
import model_testing

MAGS_BINS = deb_example.mags_bins
NUM_EXT_INPUTS = len(deb_example.extra_features_and_defaults)

# We can now specify paths to train/val/test datasets separately for greater flexibility.
TRAINSET_NAME = "formal-training-dataset/"
DATASET_DIR = Path(".") / "datasets" / TRAINSET_NAME
TRAINSET_DIR = DATASET_DIR / "training"
VALIDSET_DIR = DATASET_DIR / "validation"
TESTSET_DIR = DATASET_DIR / "testing"
MODEL_FILE_NAME = "cnn_ext_model"
MODEL_NAME = "CNN-Ext-Estimator-New"
SAVE_DIR = Path(".") / "drop/"
PLOTS_DIR = SAVE_DIR / "plots"

# Formal testset is a currated set of test systems for formal testing across models.
# It's the dataset used for the testing and reports of test_estimator.
FORMAL_TESTSET_DIR = Path(".") / "datasets/formal-test-dataset/"
FORMAL_RESULTS_DIR = SAVE_DIR / f"results/{MODEL_NAME}/{TRAINSET_NAME}/{deb_example.pub_mags_key}/"

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
batch_size, buffer_size = 0, 0
map_func = deb_example.create_map_func(noise_stddev=lambda: 0.005,
                                       roll_steps=lambda: tf.random.uniform([], -9, 10, tf.int32))
for ds_ix, (label, set_dir) in enumerate([("training", TRAINSET_DIR),
                                          ("valiation", VALIDSET_DIR),
                                          ("testing", TESTSET_DIR)]):
    # Don't set up any mappings/shuffle/batch yet as we want to count the contents first
    files = list(set_dir.glob("**/*.tfrecord"))
    datasets[ds_ix] = tf.data.TFRecordDataset(files, num_parallel_reads=100)
    counts[ds_ix] = datasets[ds_ix].reduce(0, lambda count, _: count+1).numpy()
    print(f"Found {counts[ds_ix]:,} {label} instances spread over",
          f"{len(files)} tfrecord dataset file(s) within '{set_dir}'.")

    # Now, having counted them, we can set up the full dataset pipelines
    if ds_ix == 0:
        batch_size = round(counts[0] * BATCH_FRACTION)
        buffer_size = min(MAX_BUFFER_SIZE, counts[0])
        datasets[ds_ix] = datasets[ds_ix] \
                            .shuffle(buffer_size, SEED, reshuffle_each_iteration=True) \
                                .map(map_func) \
                                    .batch(batch_size) \
                                        .prefetch(1)
    else:
        datasets[ds_ix] = datasets[ds_ix].map(map_func).batch(batch_size)


# -----------------------------------------------------------
# Define the model
# -----------------------------------------------------------
print("\nDefining the multiple-input/output CNN model.")

def build_cnn_layers(tensor):
    """
    re-dimension each instance's Mags from 1d [#bins, 1] to 2D [bins, features] of dimension [8, 64]
    """
    tensor = modelling.conv1d_layers(tensor, 2, 64, 8, 4, CNN_PADDING, CNN_ACTIVATE, "CNN-1.")
    tensor = modelling.conv1d_layers(tensor, 3, 64, 4, 2, CNN_PADDING, CNN_ACTIVATE, "CNN-2.")
    return tensor

def build_dnn_layers(tensor):
    """
    The Dropout layers randomly "drop" (set to 0) the ratio of inputs each training iteration.
    Note: undermines comparison of training & validation loss as validation results w/o dropout
    """
    tensor = modelling.hidden_layers(tensor, DNN_NUM_FULL_LAYERS, 256, DNN_INITIALIZER,
                                     DNN_ACTIVATE, DNN_DROPOUT_RATE, ("Hidden-", "Dropout-"))
    # "Buffer" between the DNN+Dropout and the output layer; this non-dropout NN layer
    # consistently gives a small, but significant improvement to the trained loss.
    tensor = modelling.hidden_layers(tensor, 1, 128, DNN_INITIALIZER, DNN_ACTIVATE, 0, ("Taper-", ))
    return tensor

model = modelling.build_mags_ext_model(
    build_mags_layers=build_cnn_layers,
    build_dnn_layers=build_dnn_layers,
    name=MODEL_NAME)
model.summary()

try:
    # Can only get this working specific pydot (1.4) & graphviz (8.0) conda packages.
    # With pip I can't get graphviz beyond 0.2.0 which leads to pydot errors here.
    # At least with the try block I can degrade gracefully.
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_model(model, to_file=PLOTS_DIR / f"{MODEL_FILE_NAME}.png",
               show_layer_names=True, show_shapes=True, show_layer_activations=True,
               show_dtype=False, show_trainable=False, rankdir="LR", dpi=300)
except ImportError:
    print("Unable to plot_model() without pydot and/or graphviz.")


# -----------------------------------------------------------
# Build & Train the model
# -----------------------------------------------------------
print("Building the model.")
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

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
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f"{MODEL_FILE_NAME}_learning_curves.png", dpi=300)
except tf.errors.InvalidArgumentError as exc:
    if ("lc" in exc.message or "mags" in exc.message) and "Can't parse serialized" in exc.message:
        msg = exc.message + "\n*** Probable cause: incompatible serialized mags feature length. ***"
        raise tf.errors.InvalidArgumentError(exc.node_def, exc.op, msg, exc.args) from exc

print(f"\nEvaluating the model on {counts[2]} test instances.")
model.evaluate(datasets[2], verbose=1)

# Save the newly trained model
model_save_file = SAVE_DIR / f"{MODEL_FILE_NAME}.keras"
# custom_attrs = {
#     "training_dataset": TRAINSET_NAME,
#     "training_instances": counts[0],
#     "training_cuda_devices": tf.config.experimental.list_physical_devices('GPU'),
#     "mags_wrap_phase": MAGS_WRAP_PHASE,
# }
modelling.save_model(model_save_file, model)
print(f"\nSaved model '{MODEL_NAME}' to: {model_save_file}")

# -----------------------------------------------------------
# Tests the newly saved model, within an Estimator, against a dataset of real systems.
# -----------------------------------------------------------
print("\n *** Running formal test against real data ***")
FORMAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
model_testing.test_with_estimator(model_save_file, FORMAL_TESTSET_DIR, FORMAL_RESULTS_DIR)
