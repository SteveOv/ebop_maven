""" Contains functions for building and persisting ML models """
from typing import Union, List, Tuple, Iterable
from pathlib import Path

from keras import layers, models

def append_conv1d_layers(previous_layer: layers.Layer,
                         num_layers: int=1,
                         filters: Union[int, List[int]]=64,
                         kernel_size: Union[int, List[int]]=8,
                         strides: Union[int, List[int]]=2,
                         padding: Union[str, List[str]]="same",
                         activation: Union[any, List[any]]="ReLU",
                         name_prefix: str="CNN-"):
    """
    Appends the requested set of Conv1D layers.
    
    The filters, kernel_size, strides, padding and activation arguments can be
    a List of values, one per layer, or a single values used for every layer.
    
    :previous_layer: the existing layer to append to
    :num_layers: number of Conv1D layers to create
    :filters: the filters value of each layer
    :kernel_size: the kernel_size of each layer
    :strides: the strides value of each layer
    :padding: the padding value of each layer
    :activation: the activation value of each layer
    :name_prefix: the text to prefix the indexed layer name
    :returns: the final new layer appended
    """
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
        previous_layer = layers.Conv1D(filters=filters[ix],
                                       kernel_size=kernel_size[ix],
                                       strides=strides[ix],
                                       padding=padding[ix],
                                       activation=activation[ix],
                                       name=f"{name_prefix}{ix}")(previous_layer)
    return previous_layer


def append_concatenate_layer(previous_layers: Iterable[layers.Layer],
                             axis: int = -1,
                             name: str = "Concatenate") -> layers.Layer:
    """
    Appends a new concatenate layer which flattens and joins previous_layers

    :previous_layers: the existing layers to concatenate
    :axis: the axis along which to concatenate (defaults to the last axis)
    :name: the name to give the Concatenate layer
    :returns: the newly created Concatenate layer
    """
    return layers.Concatenate(axis, name=name) \
                ([layers.Flatten(name=f"Flat-{ix}")(pl) for (ix, pl) in enumerate(previous_layers)])


def append_hidden_layers(previous_layer: layers.Layer,
                         num_layers: int=1,
                         units: Union[int, List[int]]=256,
                         activation: Union[any, List[any]]=None,
                         kernel_initializer: Union[str, List[str]]="glorot_uniform",
                         dropout_rate: Union[float, List[float]]=0,
                         name_prefix: Tuple[str, str]=("Hidden-", "Dropout-")) -> layers.Layer:
    """
    Appends the requested set of hidden Dense layers with optional accompanying Dropout layers.

    The units, activation, kernel_initializer and dropout_rate arguments can be
    a List of values, one per layer, or a single values used for every layer.
    
    :previous_layer: the existing layer to append to
    :num_layers: number of Dense+Dropout layer pairs to append
    :units: the number of neurons in new each layer
    :activation: the activation function for each new layer
    :kernel_initializer: the initializer for each new layer
    :dropout_rate: the fraction of the units to drop for each layer. No dropout layer when set to 0
    :name_prefix: the text to prefix the indexed layer name
    :returns: the final new layer appended
    """
    if not isinstance(units, List):
        units = [units] * num_layers
    if not isinstance(activation, List):
        activation = [activation] * num_layers
    if not isinstance(kernel_initializer, List):
        kernel_initializer = [kernel_initializer] * num_layers
    if not isinstance(dropout_rate, List):
        dropout_rate = [dropout_rate] * num_layers

    # Expected failure if any list isn't num_layers long
    for ix in range(num_layers):
        previous_layer = layers.Dense(units[ix],
                                      activation=activation[ix],
                                      kernel_initializer=kernel_initializer[ix],
                                      name=f"{name_prefix[0]}{ix}")(previous_layer)
        if dropout_rate[ix]:
            previous_layer = layers.Dropout(dropout_rate[ix],
                                            name=f"{name_prefix[1]}{ix}")(previous_layer)
    return previous_layer


def append_output_layer(previous_layer: layers.Layer, units: int):
    """
    Appends a new Output layer with the linear activation

    :previous_layer: the existing layer to append to
    :name: the name to give the layer
    :returns: the newly created output layer
    """
    return layers.Dense(units, activation="linear", name="Output")(previous_layer)


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
