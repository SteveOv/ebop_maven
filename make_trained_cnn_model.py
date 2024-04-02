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
from keras import models, layers, optimizers, callbacks

from ebop_maven import tensorflow_models
from ebop_maven.libs import deb_example


LC_WRAP_PHASE = 0.75            # Control the shape and default roll of LC feature
LC_BINS = deb_example.description["lc"].shape[0]
# These control the data augmentation layers early in the LC/CNN branch of the model
LC_TRAINING_NOISE = True        # Best to use this on a "un-noised" dataset
LC_TRAINING_ROLL = False

# We can now specify paths to train/val/test datasets separately for greater flexibility.
TRAINSET_NAME = f"formal-trainset/{LC_BINS}/wm-{LC_WRAP_PHASE}"
DATASET_DIR = Path(".") / "datasets" / TRAINSET_NAME
TRAINSET_DIR = DATASET_DIR / "training"
VALIDSET_DIR = DATASET_DIR / "validation"
TESTSET_DIR = DATASET_DIR / "testing"
MODEL_FILE_NAME = "cnn_ext_model"
MODEL_NAME = "CNN-Ext-Estimator-New"
SAVE_DIR = Path(".") / "models/"
PLOTS_DIR = SAVE_DIR / "plots"

# Formal testset is a currated set of test systems for formal testing across models.
# It's the dataset used for the testing and reports of test_estimator.
FORMAL_TESTSET_DIR = Path(".") / "datasets" / f"formal-test-dataset/{LC_BINS}/wm-{LC_WRAP_PHASE}/"
FORMAL_RESULTS_DIR = SAVE_DIR / "results" / MODEL_NAME / TRAINSET_NAME

NUMBER_FULL_HIDDEN_LAYERS = 2   # Number of full width hidden layers (with associated dropout)
TRAINING_EPOCHS = 10           # Set high if we're using early stopping
BATCH_FRACTION = 0.001          # larger -> quicker training per epoch but more to converge
MAX_BUFFER_SIZE = 20000000      # Size of Dataset shuffle buffer (in instances)
EARLY_STOPPING_PATIENCE = 7     # Number of epochs w/o improvement before stopping

ENFORCE_REPEATABILITY = True    # If true, avoid GPU/CUDA cores for repeatable results
SEED = 42                       # Standard random seed ensures repeatable randomization
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

LOSS = ["mae"] * len(deb_example.label_names)
METRICS = [["mse"]]
PADDING = "same"

# ReLU is widely used default for CNN/DNNs.
# Otherwise, may need to specify each layer separately as dims different.
# LeakyReLU addresses issue of dead neurons & PReLU similar but trains alpha param
CNN_ACTIVATE = layers.ReLU()
DNN_ACTIVATE = layers.LeakyReLU()

# For the dense layers: "glorot_uniform" (def) "he_normal", "he_uniform" (he_ goes well with ReLU)
KERNEL_INITIALIZER = "he_uniform"

# Decay will reduce learning rate ever more slowly over time (see Learning Rate/Power Scheduling)
OPTIMIZER = optimizers.nadam_v2.Nadam(learning_rate=5e-4, decay=1e-4)

# Regularization: fraction of DNN neurons to drop on each training step
DROPOUT_RATE=0.5

print(f"""tensorflow v{tf.__version__}
tensorboard v{tensorboard.__version__}
keras v{tf.keras.__version__}""")

if ENFORCE_REPEATABILITY:
    # Extreme, but it stops TensorFlow/Keras from using (even seeing) the GPU.
    # Slows training down massively (by 3-4 times) but should avoid GPU memory
    # constraints! Necessary if repeatable results are required (Keras advises
    # that out of order processing within GPU/CUDA can lead to varying results).
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(f"Found {len(tf.config.experimental.list_physical_devices('GPU'))} GPU(s)\n")


# -----------------------------------------------------------
# Set up the training/validation/test datasets
# -----------------------------------------------------------
print("Picking up training/validation/test datasets.")
datasets = [tf.data.TFRecordDataset] * 3
counts = [int] * 3
batch_size, buffer_size = 0, 0
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
                                .map(deb_example.map_parse_deb_example) \
                                    .batch(batch_size) \
                                        .prefetch(1)
    else:
        datasets[ds_ix] = datasets[ds_ix].map(deb_example.map_parse_deb_example).batch(batch_size)


# -----------------------------------------------------------
# Define the model
# -----------------------------------------------------------
print("\nDefining the multiple-input/output CNN model.")

# Input for the Light-curve (Timeseries data) via a CNN.
cnn = lc_input = layers.Input(shape=(LC_BINS, 1), name="LC-Input")

# Input LC augmentations; Gaussian noise and roll/rotate (misaligned eclipses).
# Set these up to have no effect, even when training is on, so that they
# will not affect MC Dropout style predictions. During real training we can
# use callbacks to set and/or vary the scale of the effect.
cnn = layers.GaussianNoise(stddev=0, name="Training-Noise")(cnn)
cnn = tensorflow_models.Roll1D(roll_by=0, name="Training-Roll")(cnn)

# 1D conv layers re-dimension each LS instances from 1d timeseries #bins long
# into 2D [timeseries, features] of dimension [8, 64]
cnn = tensorflow_models.conv1d_layers(cnn, num_layers=2, filters=64, kernel_size=8, strides=4,
                                    padding=PADDING, activation=CNN_ACTIVATE, name_prefix="CNN-1.")
cnn = tensorflow_models.conv1d_layers(cnn, num_layers=3, filters=64, kernel_size=4, strides=2,
                                    padding=PADDING, activation=CNN_ACTIVATE, name_prefix="CNN-2.")

# Input for the Extra features
ext_input = layers.Input(shape=(len(deb_example.extra_features_and_defaults), 1), name="Ext-Input")

# Combine the separate input paths for the LC (RNN/LSTM) & EXT Features to the DNN
dnn = layers.Concatenate(axis=1, name="DNN-Combined-Input")([
            layers.Flatten(name="LC-Reshape")(cnn),
            layers.Flatten(name="Ext-Reshape")(ext_input)])

# The Dropout layers are a regularization mechanism (they combat overfitting).
# They randomly "drop" (set to 0) the ratio of inputs each training iteration.
# Note: undermines any comparison of training & validation loss
#       as validation results are calculated w/o dropout
dnn = tensorflow_models.hidden_layers(dnn, NUMBER_FULL_HIDDEN_LAYERS, units=256,
                                      kernel_initializer=KERNEL_INITIALIZER,
                                      activation=DNN_ACTIVATE,
                                      dropout_rate=DROPOUT_RATE,
                                      name_prefix=("Hidden-", "Dropout-"))

# "Buffer" between the DNN+Dropout and the output layer; this non-dropout NN layer
# consistently gives a small, but significant improvement to the trained loss.
dnn = tensorflow_models.hidden_layers(dnn, 1, 128, kernel_initializer=KERNEL_INITIALIZER,
                                      activation=DNN_ACTIVATE, dropout_rate=0,
                                      name_prefix=("Taper", None))

# Sets up the output predicted values
dnn = layers.Dense(len(deb_example.label_names), activation="linear", name="Output")(dnn)

model = models.Model(inputs=[lc_input, ext_input], outputs=dnn, name=MODEL_NAME)
model.summary()

# TODO: need to get compatible pydot & graphviz version to install through pip
# PLOTS_DIR.mkdir(parents=True, exist_ok=True)
# plot_model(model, to_file=f"./models/saves/plots/{MODEL_FILE_NAME}.png",
#            show_shapes=True, show_layer_names=True, dpi=300)


# -----------------------------------------------------------
# Build & Train the model
# -----------------------------------------------------------
print("Building the model.")
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

CALLBACKS = [
    # To use tensorboard make sure deblcest conda env is active then run
    # $ tensorboard --port 6006 --logdir ./logs
    # Then start a browser and head to http://localhost:6006
    #callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True),
    callbacks.EarlyStopping("val_loss", restore_best_weights=True,
                            patience=EARLY_STOPPING_PATIENCE, verbose=1)
]

if LC_TRAINING_NOISE:
    CALLBACKS += [tensorflow_models.SetLayerAttribute(model.get_layer("Training-Noise"), "stddev",
                                                  on_train_begin=lambda: 0.005)]
if LC_TRAINING_ROLL:
    CALLBACKS += [tensorflow_models.SetLayerAttribute(model.get_layer("Training-Roll"), "roll_by",
                                                  on_batch_begin=lambda: np.random.randint(-3, 4))]

print(f"\nTraining the model on {counts[0]} training and {counts[1]} validation",
      f"instances, with a further {counts[2]} instances held back for test.")
try:
    history = model.fit(x = datasets[0],    # pylint: disable=invalid-name
                        epochs = TRAINING_EPOCHS,
                        callbacks = CALLBACKS,
                        validation_data = datasets[1],
                        max_queue_size = 50,
                        use_multiprocessing = True,
                        workers = 2)
    # Plot the learning curves
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f"{MODEL_FILE_NAME}_learning_curves.png", dpi=300)
except tf.errors.InvalidArgumentError as exc:
    if "lc" in exc.message and "Can't parse serialized Example" in exc.message:
        msg = exc.message + "\n*** Probable cause: incompatible serialized lc length. ***"
        raise tf.errors.InvalidArgumentError(exc.node_def, exc.op, msg, exc.args) from exc

print(f"\nEvaluating the model on {counts[2]} test instances.")
model.evaluate(datasets[2], use_multiprocessing=True, verbose=1)

# Save the newly trained model
model_save_file = SAVE_DIR / f"{MODEL_FILE_NAME}.h5"
custom_attrs = {
    "training_dataset": TRAINSET_NAME,
    "training_instances": counts[0],
    "training_cuda_devices": tf.config.experimental.list_physical_devices('GPU'),
    "lc_training_noise": LC_TRAINING_NOISE,
    "lc_training_roll": LC_TRAINING_ROLL,
    "lc_wrap_phase": LC_WRAP_PHASE,
}
tensorflow_models.save_model(model_save_file, model, custom_attrs)
print(f"\nSaved model '{MODEL_NAME}' to: {model_save_file}")

# -----------------------------------------------------------
# Tests the newly saved model, within an Estimator, against a dataset of real systems.
# -----------------------------------------------------------
print("\n *** Running formal test against real data ***")
FORMAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#test_estimator(model_save_file, FORMAL_TESTSET_DIR, FORMAL_RESULTS_DIR)
