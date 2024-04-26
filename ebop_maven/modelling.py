""" Contains functions for building and persisting ML functional models """
from typing import Union, List, Dict, Tuple, Callable
from pathlib import Path

from keras import layers, models, KerasTensor
from keras.src.layers.pooling.base_pooling import BasePooling

from .libs import deb_example


class OutputLayer(layers.Dense):
    """ A custom OutputLayer which can store metadata about the labels being predicted. """
    # pylint: disable=abstract-method, too-many-ancestors, too-many-arguments

    @property
    def label_names_and_scales(self) -> Dict[str, float]:
        """ The names and any scaling which needs to be undone. """
        return self._label_names_and_scales

    def __init__(self,
                 units:int,
                 kernel_initializer: str="he_uniform",
                 activation: str="linear",
                 name: str="Output",
                 label_names_and_scales: Dict[str, float] = None,
                 **kwargs):
        """
        Initializes a new OutputLayer

        :units: the number of predicting values
        :kernel_initializer: dictates how the neurons are initialized
        :activation: the activation function to use
        :name: the name of the layer
        :label_names_and_scales: dictionary giving the output labels' names and any scaling applied
        """
        if label_names_and_scales:
            num_name = len(label_names_and_scales)
            if num_name != units:
                raise ValueError(f"len(label_names_and_scales)!={units} units. Actually {num_name}")
        self._label_names_and_scales = label_names_and_scales
        super().__init__(units,
                         kernel_initializer=kernel_initializer,
                         activation=activation,
                         name=name,
                         **kwargs)

    def get_config(self):
        """ Serializes this layer. """
        return { **super().get_config(), "label_names_and_scales": self.label_names_and_scales }


def conv1d_layers(num_layers: int=1,
                  filters: Union[int, List[int]]=64,
                  kernel_size: Union[int, List[int]]=8,
                  strides: Union[int, List[int]]=2,
                  padding: Union[str, List[str]]="same",
                  activation: Union[any, List[any]]="relu",
                  name_prefix: str="Conv-",
                  verbose: bool=True) -> KerasTensor:
    """
    Builds the requested set of Conv1D layers, returning the output tensor of the last layer built.
    The filters, kernel_size, strides, padding and activation arguments can be a List of values,
    one per layer, or a single values used for every layer.
    
    :num_layers: number of Conv1D layers to create
    :filters: the filters value of each layer
    :kernel_size: the kernel_size of each layer
    :strides: the strides value of each layer
    :padding: the padding value of each layer
    :activation: the activation value of each layer
    :name_prefix: the text to prefix the indexed layer name
    :verbose: print out info of what's happening
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

    def layers_func(input_tensor: KerasTensor) -> KerasTensor:
        # Expected failure if any list isn't num_layers long
        for ix in range(num_layers):
            name = f"{name_prefix}{ix+1}"
            output_tensor = layers.Conv1D(filters=filters[ix],
                                         kernel_size=kernel_size[ix],
                                         strides=strides[ix],
                                         padding=padding[ix],
                                         activation=activation[ix],
                                         name=name)(input_tensor)
            if verbose:
                print(f"Creating Conv1D('{name}', filters={filters[ix]},",
                      f"kernel_size={kernel_size[ix]}, strides={strides[ix]},",
                      f"padding={padding[ix]}, activation={activation[ix]})",
                      f"({input_tensor.shape}) -> {output_tensor.shape}")
            input_tensor = output_tensor
        return output_tensor
    return layers_func


def pooling_layer(pool_type: BasePooling,
                  pool_size: int=2,
                  strides: int=2,
                  name: str=None,
                  verbose: bool=True) -> KerasTensor:
    """
    Builds the requested Pooling layer, returning its output tensor.

    :pool_type: the type of layer
    :pool_size: the pool_size of the layer
    :strides: the strides value of the layer
    :name: the name of the layer
    :verbose: print out info of what's happening
    :returns: the output tensor of the new layer
    """
    def layer_func(input_tensor: KerasTensor) -> KerasTensor:
        output_tensor = pool_type(pool_size=pool_size, strides=strides, name=name)(input_tensor)
        if verbose:
            print(f"Creating {pool_type.__name__}('{name}',"
                  f"pool_size={pool_size}, strides={strides})",
                  f"({input_tensor.shape}) -> {output_tensor.shape}")
        return output_tensor
    return layer_func


def hidden_layers(num_layers: int=1,
                  units: Union[int, List[int]]=256,
                  kernel_initializer: Union[str, List[str]]="glorot_uniform",
                  activation: Union[any, List[any]]="relu",
                  dropout_rate: Union[float, List[float]]=0,
                  name_prefix: Tuple[str, str]=("Hidden-", "Dropout-"),
                  verbose: bool=True) -> KerasTensor:
    """
    Builds the requested set of hidden Dense layers with optional accompanying Dropout layers,
    returning the output tensor of the last layer built. The units, activation, kernel_initializer
    and dropout_rate arguments can be a List of values, one per layer, or a single values used for
    every layer.
    
    :num_layers: number of Dense+Dropout layer pairs to append
    :units: the number of neurons in new each layer
    :kernel_initializer: the initializer for each new layer
    :activation: the activation function for each new layer
    :dropout_rate: the fraction of the units to drop for each layer. No dropout layer when set to 0
    :name_prefix: the text to prefix the indexed layer name
    :verbose: print out info of what's happening
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

    def layers_func(input_tensor: KerasTensor) -> KerasTensor:
        # Expected failure if any list isn't num_layers long
        for ix in range(num_layers):
            name = f"{name_prefix[0]}{ix+1}"
            output_tensor = layers.Dense(units[ix],
                                         kernel_initializer=kernel_initializer[ix],
                                         activation=activation[ix],
                                         name=name)(input_tensor)
            if verbose:
                print(f"Creating Dense('{name}', units={units[ix]},",
                      f"kernel_initializer={kernel_initializer[ix]},",
                      f"activation={activation[ix]})",
                      f"({input_tensor.shape}) -> {output_tensor.shape}")

            input_tensor = output_tensor
            if dropout_rate[ix]:
                name = f"{name_prefix[1]}{ix+1}"
                output_tensor = layers.Dropout(dropout_rate[ix], name=name)(input_tensor)
                if verbose:
                    print(f"Creating Dropout('{name}', rate={dropout_rate[ix]})",
                          f"({input_tensor.shape}) -> {output_tensor.shape}")
                input_tensor = output_tensor
        return output_tensor
    return layers_func


def mags_input_layer(shape: Tuple[int, int]=(deb_example.mags_bins, 1),
                     name: str="Mags-Input",
                     verbose: bool=True) -> KerasTensor:
    """
    Builds a standard mags-feature input layer.

    :shape: the shape of the input tensors
    :name: the name of the layer
    :verbose: print out info of what's happening
    :returns: the output tensor of the input layer
    """
    if verbose:
        print(f"Creating Input('{name}', shape={shape})")
    return layers.Input(shape=shape, name=name)


def ext_input_layer(shape: Tuple[int, int]=(len(deb_example.extra_features_and_defaults), 1),
                    name: str="Ext-Input",
                    verbose: bool=True) -> KerasTensor:
    """
    Builds a standard ext-features input layer.

    :shape: the shape of the input tensors
    :name: the name of the layer
    :verbose: print out info of what's happening
    :returns: the output tensor of the input layer
    """
    if verbose:
        print(f"Creating Input('{name}', shape={shape})")
    return layers.Input(shape=shape, name=name)


def output_layer(label_names_and_scales: Dict[str, any]=None,
                 kernel_initializer: str="glorot_uniform",
                 activation: str="linear",
                 name: str="Output",
                 verbose: bool=True) -> KerasTensor:
    """
    Builds the requested output layer, returning its output tensor.

    :label_names_and_scales: the dictionary of labels we will be predicting
    :kernel_initializer: the initializer
    :activation: the activation function
    :name: the name of the layer
    :verbose: print out info of what's happening
    :returns: the output tensor of the new layer
    """
    if not label_names_and_scales:
        label_names_and_scales = deb_example.labels_and_scales
    def layer_func(input_tensor: KerasTensor) -> KerasTensor:
        units = len(label_names_and_scales)
        output_tensor = OutputLayer(units=units,
                                    kernel_initializer=kernel_initializer,
                                    activation=activation,
                                    label_names_and_scales=label_names_and_scales,
                                    name=name)(input_tensor)
        if verbose:
            print(f"Creating OutputLayer('{name}', units={units},",
                  f"kernel_initializer={kernel_initializer}, activation={activation},",
                  f"label_names_and_scales={label_names_and_scales})",
                  f"({input_tensor.shape}) -> {output_tensor.shape}")
        return output_tensor
    return layer_func


def build_mags_ext_model(
        name: str="Mags-Ext-Model",
        mags_input: KerasTensor=mags_input_layer(verbose=False),
        ext_input: KerasTensor=ext_input_layer(verbose=False),
        mags_layers: List[Callable[[KerasTensor], KerasTensor]]=None,
        ext_layers: List[Callable[[KerasTensor], KerasTensor]]=None,
        dnn_layers: List[Callable[[KerasTensor], KerasTensor]]=None,
        output: Callable[[KerasTensor], KerasTensor]=output_layer(verbose=False),
        post_build_step: Callable[[models.Model], None]=None,
        verbose: bool=True
    ) -> models.Model:
    """
    Builds a multiple input model with separate Mags-Feature and Extra-Features inputs.
    These are concatenated and a Deep Neural Network is appended. Finally an output layer
    is appended before the final Model is returned.

    :name: the name of the new model
    :mags_input: the phase-folded lightcurve magnitudes feature input
    :ext_input: the extra features input
    :mags_layers: list of layers which process the mags input tensor
    :ext_layers: list of layers which process the ext_features input tensor
    :dnn_layers: list of neural network layers for processing the concatenated
    outputs of the mags and ext layers
    :output: the output layer for publishing the output of the dnn_layers
    :post_build_step: optional hook to interact with the newly built model
    :verbose: print out info of what's happening
    :returns: the new model
    """
    # pylint: disable=too-many-arguments

    # Build up the model by building the two input branches, concatenating them, passing their
    # combined output to a DNN and then passing the DNN output to a set of output neurons.
    mags_tensor = mags_input
    if mags_layers:
        if not isinstance(mags_layers, List):
            mags_layers = [mags_layers]
        for buildit in mags_layers:
            mags_tensor = buildit(mags_tensor)

    ext_tensor = ext_input
    if ext_layers:
        if not isinstance(ext_layers, List):
            ext_layers = [ext_layers]
        for buildit in ext_layers:
            ext_tensor = buildit(ext_tensor)

    output_tensor = layers.Concatenate(axis=1, name="DNN-Input")([
                    layers.Flatten(name="Mags-Reshape")(mags_tensor),
                    layers.Flatten(name="Ext-Reshape")(ext_tensor)
                ])
    if verbose:
        print( "Creating Concatenate('DNN-Input', axis=1)",
              f"([Flatten('Mags-Reshape')({mags_tensor.shape}),"
              f" Flatten('Ext-Reshape')({ext_tensor.shape})]) -> {output_tensor.shape}")

    if dnn_layers:
        if not isinstance(dnn_layers, List):
            dnn_layers = [dnn_layers]
        for buildit in dnn_layers:
            output_tensor = buildit(output_tensor)

    output_tensor = output(output_tensor)
    model = models.Model(inputs=[mags_input, ext_input], outputs=output_tensor, name=name)

    if post_build_step:
        if verbose:
            print(f"Calling out to the post_build_step: {post_build_step}")
        post_build_step(model)

    return model


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
                                 "OutputLayer": OutputLayer,
                                 # I shouldn't *have* to do this; fails to deserialize otherwise!
                                 "ReLU": layers.ReLU,
                                 "LeakyReLU": layers.LeakyReLU,
                                 "ELU": layers.ELU,
                            })
