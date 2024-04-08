""" Contains functions for building and persisting ML models """
from typing import Union, List, Tuple, Callable
from pathlib import Path

from keras import layers, models, KerasTensor

from .libs import deb_example

def conv1d_layers(last_tensor: KerasTensor,
                  num_layers: int=1,
                  filters: Union[int, List[int]]=64,
                  kernel_size: Union[int, List[int]]=8,
                  strides: Union[int, List[int]]=2,
                  padding: Union[str, List[str]]="same",
                  activation: Union[any, List[any]]="relu",
                  name_prefix: str="CNN-") -> KerasTensor:
    """
    Builds the requested set of Conv1D layers, returning the output tensor of the last layer built.
    The filters, kernel_size, strides, padding and activation arguments can be a List of values,
    one per layer, or a single values used for every layer.
    
    :last_tensor: the output tensor from the preceding layer
    :num_layers: number of Conv1D layers to create
    :filters: the filters value of each layer
    :kernel_size: the kernel_size of each layer
    :strides: the strides value of each layer
    :padding: the padding value of each layer
    :activation: the activation value of each layer
    :name_prefix: the text to prefix the indexed layer name
    :returns: the output tensor of the last new layer
    """
    # pylint: disable=too-many-arguments
    if not isinstance(filters, List):
        filters = [filters] * num_layers
    if not isinstance(kernel_size, List):
        kernel_size = [kernel_size] * num_layers
    if not isinstance(strides, List):
        strides = [strides] * num_layers
    if not isinstance(padding, List):
        padding = [padding] * num_layers
    if not isinstance(activation, List):
        activation = [activation] * num_layers

    # Expected failure if any list isn't num_layers long
    for ix in range(num_layers):
        last_tensor = layers.Conv1D(filters=filters[ix],
                                    kernel_size=kernel_size[ix],
                                    strides=strides[ix],
                                    padding=padding[ix],
                                    activation=activation[ix],
                                    name=f"{name_prefix}{ix}")(last_tensor)
    return last_tensor


def hidden_layers(last_tensor: KerasTensor,
                  num_layers: int=1,
                  units: Union[int, List[int]]=256,
                  kernel_initializer: Union[str, List[str]]="glorot_uniform",
                  activation: Union[any, List[any]]="relu",
                  dropout_rate: Union[float, List[float]]=0,
                  name_prefix: Tuple[str, str]=("Hidden-", "Dropout-")) -> KerasTensor:
    """
    Builds the requested set of hidden Dense layers with optional accompanying Dropout layers,
    returning the output tensor of the last layer built. The units, activation, kernel_initializer
    and dropout_rate arguments can be a List of values, one per layer, or a single values used for
    every layer.
    
    :last_tensor: the output tensor from the preceding layer
    :num_layers: number of Dense+Dropout layer pairs to append
    :units: the number of neurons in new each layer
    :kernel_initializer: the initializer for each new layer
    :activation: the activation function for each new layer
    :dropout_rate: the fraction of the units to drop for each layer. No dropout layer when set to 0
    :name_prefix: the text to prefix the indexed layer name
    :returns: the output tensor of the last new layer
    """
    # pylint: disable=too-many-arguments
    if not isinstance(units, List):
        units = [units] * num_layers
    if not isinstance(kernel_initializer, List):
        kernel_initializer = [kernel_initializer] * num_layers
    if not isinstance(activation, List):
        activation = [activation] * num_layers
    if not isinstance(dropout_rate, List):
        dropout_rate = [dropout_rate] * num_layers

    # Expected failure if any list isn't num_layers long
    for ix in range(num_layers):
        last_tensor = layers.Dense(units[ix],
                                   kernel_initializer=kernel_initializer[ix],
                                   activation=activation[ix],
                                   name=f"{name_prefix[0]}{ix}")(last_tensor)
        if dropout_rate[ix]:
            last_tensor = layers.Dropout(dropout_rate[ix],
                                         name=f"{name_prefix[1]}{ix}")(last_tensor)
    return last_tensor


def mags_input_layer(shape: Tuple[int, int]=(deb_example.mags_bins, 1),
                     name: str="Mags-Input") -> KerasTensor:
    """
    Builds a standard mags-feature input layer.

    :shape: the shape of the input tensors
    :name: the name of the layer
    :returns: the output tensor of the input layer
    """
    return layers.Input(shape=shape, name=name)


def ext_input_layer(shape: Tuple[int, int]=(len(deb_example.extra_features_and_defaults), 1),
                    name: str="Ext-Input") -> KerasTensor:
    """
    Builds a standard ext-features input layer.

    :shape: the shape of the input tensors
    :name: the name of the layer
    :returns: the output tensor of the input layer
    """
    return layers.Input(shape=shape, name=name)


def output_layer(last_tensor: KerasTensor,
                 units: int=len(deb_example.label_names),
                 kernel_initializer: str="glorot_uniform",
                 activation: str="linear",
                 name: str="Output") -> KerasTensor:
    """
    Builds the requested output layer, returning its output tensor.

    :last_tensor: the output tensor from the preceding layer
    :units: the number of output neurons
    :kernel_initializer: the initializer
    :activation: the activation function
    :name: the name of the layer
    :returns: the output tensor of the new layer
    """
    return layers.Dense(units,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        name=name)(last_tensor)

def empty_layer(last_tensor: KerasTensor) -> KerasTensor:
    """
    Do not build any layers ... just return the output tensor of the preceding layer.
    """
    return last_tensor


def build_mags_ext_model(
        mags_input: KerasTensor=mags_input_layer(),
        ext_input: KerasTensor=ext_input_layer(),
        build_mags_layers: Callable[[KerasTensor], KerasTensor]=empty_layer,
        build_ext_layers: Callable[[KerasTensor], KerasTensor]=empty_layer,
        build_dnn_layers: Callable[[KerasTensor], KerasTensor]=empty_layer,
        build_output_layer: Callable[[KerasTensor], KerasTensor]=output_layer,
        name: str="Mags-Ext-Model"
    ) -> models.Model:
    """
    Builds a multiple input model with separate Mags-Feature and Extra-Features inputs.
    These are concatenated and a Deep Neural Network is appended. Finally an output layer
    is appended before the final Model is returned.

    :mags_input: the phase-folded lightcurve magnitudes feature input
    :ext_input: the extra features input
    :build_mags_layers: function which builds the magnitudes input branch,
        with the output tensor of mags_input as its input
    :build_ext_layers: function which builds the extra features input branch,
        with the output tensor of ext_input as its input
    :build_dnn_layers: function which builds the deep neural network,
        with the output tensor of the concatenated mags & ext input branches as its input
    :build_output_layer: function which build the layer of output neurons,
        with the output tensor of the dnn layers as its input
    :name: the name of the new model
    :returns: the new model
    """
    # pylint: disable=too-many-arguments

    # Build up the model by building the two input branches, concatenating them, passing their
    # combined output to a DNN and then passing the DNN output to a set of output neurons.
    output_tensor = build_output_layer(
        build_dnn_layers(
            layers.Concatenate(axis=1, name="DNN-Input")([
                layers.Flatten(name="Mags-Reshape")(build_mags_layers(mags_input)),
                layers.Flatten(name="Ext-Reshape")(build_ext_layers(ext_input))
            ])))

    return models.Model(inputs=[mags_input, ext_input], outputs=output_tensor, name=name)


def save_model(file_name: Path,
               model: models.Model):
    """
    Save the Model, overwriting existing, to the indicated file

    TODO: try to regain the functionality of saving arbitrary metadata with the model

    :file_name: the file name of the file to save to - will be overwritten
    :model: the model to save
    """
    file_name.parent.mkdir(parents=True, exist_ok=True)
    models.save_model(model, file_name, overwrite=True)


def load_model(file_name: Path) -> models.Model:
    """
    Loads the Model and while handling the need to register the custom objects.

    TODO: try to regain the functionality of loading saved metadata from the model

    :file_name: the saved model file to load
    """
    return models.load_model(file_name,
                             custom_objects={
                                 # I shouldn't *have* to do this; fails to deserialize otherwise!
                                 "ReLU": layers.ReLU,
                                 "LeakyReLU": layers.LeakyReLU
                            })
