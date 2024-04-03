""" Contains functions for building ML models """
from typing import Union, List, Tuple, Callable
from pathlib import Path

import tensorflow as tf
from tensorflow.python.keras.utils import control_flow_util # pylint: disable=no-name-in-module
from keras import layers, callbacks, models, saving


def conv1d_layers(previous_layer: layers.Layer,
                  num_layers: int=1,
                  filters: Union[int, List[int]]=64,
                  kernel_size: Union[int, List[int]]=8,
                  strides: Union[int, List[int]]=2,
                  padding: Union[str, List[str]]="same",
                  activation: Union[any, List[any]]="ReLU",
                  name_prefix: str="CNN-"):
    """
    Create and append the requested Conv1D layers.
    
    The filters, kernel_size, strides, padding and activation arguments can be
    a List of values, one per layer, or a single values used for each layer.
    
    :previous_layer: the existing layer to append to
    :num_layers: number of Conv1D layers to create
    :filters: the filters value of each layer
    :kernel_size: the kernel_size of each layer
    :strides: the strides value of each layer
    :padding: the padding value of each layer
    :activation: the activation value of each layer
    :name_prefix: the text to prefix the indexed layer name
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


def hidden_layers(previous_layer: layers.Layer,
                  num_layers: int=1,
                  units: Union[int, List[int]]=256,
                  activation: Union[any, List[any]]=None,
                  kernel_initializer: Union[str, List[str]]="glorot_uniform",
                  dropout_rate: Union[float, List[float]]=0,
                  name_prefix: Tuple[str, str]=("Hidden-", "Dropout-")) -> layers.Layer:
    """
    Creates a set of hidden Dense layers with optional accompanying Dropout layers.
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
                                 "Roll1D": Roll1D,
                                 # I shouldn't *have* to do this; fails to deserialize otherwise!
                                 "ReLU": layers.ReLU,
                                 "LeakyReLU": layers.LeakyReLU
                            })


@saving.register_keras_serializable()
class Roll1D(layers.Layer): # pylint: disable=abstract-method
    """
    Layer which rolls 1D input data by a fixed amount of steps controlled by
    the roll_by attribute. Negative value for roll_by will indicate a 'left'
    roll and a positive value a 'right' roll.
    """

    def __init__(self, roll_by: int=0, **kwargs):
        """
        Layer which rolls 1D input data by a fixed number of datapoints.

        :roll_by: the number of datapoints to roll the data by -
        negative indicates to the left 'left' & positive to the 'right'.
        """
        super().__init__(**kwargs)
        self.roll_by = roll_by

    def call(self, inputs, training=False): # pylint: disable=arguments-differ
        if self.roll_by == 0 or not training:
            return inputs

        # Input tensor(s) to hold batch of 1+ 1d input features shape [#insts, feature-to-roll, 1]
        inputs = tf.convert_to_tensor(inputs)
        original_shape = inputs.shape
        unbatched = inputs.shape.rank == 2
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        def roll_data():
            return tf.roll(inputs, [self.roll_by], axis=[1])
        outputs = control_flow_util.smart_cond(training, roll_data, lambda: inputs)

        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        outputs.set_shape(original_shape)
        return outputs

    def get_config(self):
        return { **super().get_config(), "roll_by": self.roll_by }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-arguments
class LayerLambdaCallback(callbacks.Callback):
    """ 
    Callback for creating simple, custom callbacks on-the-fly.
    Similar to the Keras LambdaCallback class except here the anonymous
    functions also have a reference to a model layer supplied when initialized.

    This callback is constructed with anonymous functions that will be called
    at the appropriate time (during `Model.{fit | evaluate | predict}`).
    Note that the callbacks expects positional arguments, as:

    - `on_epoch_begin` and `on_epoch_end` expect arguments: layer, epoch, logs
    - `on_batch_begin` and `on_batch_end` expect arguments: layer, batch, logs
    - `on_train_begin` and `on_train_end` expect arguments: layer, logs

    on_epoch_begin: called at the beginning of every epoch.
    on_epoch_end: called at the end of every epoch.
    on_batch_begin: called at the beginning of every batch.
    on_batch_end: called at the end of every batch.
    on_train_begin: called at the beginning of model training.
    on_train_end: called at the end of model training.
    """
    def __init__(self,
                 layer: layers.Layer,
                 on_train_begin: Callable[[layers.Layer, any], None] = None,
                 on_train_end: Callable[[layers.Layer, any], None] = None,
                 on_epoch_begin: Callable[[layers.Layer, any, any], None] = None,
                 on_epoch_end: Callable[[layers.Layer, any, any], None] = None,
                 on_batch_begin: Callable[[layers.Layer, any, any], None] = None,
                 on_batch_end: Callable[[layers.Layer, any, any], None] = None):
        """
        Initializes a new LayerLambdaCallback

        - `on_epoch_begin` and `on_epoch_end` expect arguments: layer, epoch, logs
        - `on_batch_begin` and `on_batch_end` expect arguments: layer, batch, logs
        - `on_train_begin` and `on_train_end` expect arguments: layer, logs

        :layer: the layer to pass to each anonymous function
        :on_epoch_begin: called at the beginning of every epoch.
        :on_epoch_end: called at the end of every epoch.
        :on_batch_begin: called at the beginning of every batch.
        :on_batch_end: called at the end of every batch.
        :on_train_begin: called at the beginning of model training.
        :on_train_end: called at the end of model training.
        """
        super().__init__()

        if on_train_begin:
            self.on_train_begin = lambda logs: on_train_begin(layer, logs)
        if on_train_end:
            self.on_train_end = lambda logs: on_train_end(layer, logs)
        if on_epoch_begin is not None:
            self.on_epoch_begin = lambda epoch, logs: on_epoch_begin(layer, epoch, logs)
        if on_epoch_end is not None:
            self.on_epoch_end = lambda epoch, logs: on_epoch_end(layer, epoch, logs)
        if on_batch_begin is not None:
            self.on_batch_begin = lambda batch, logs: on_batch_begin(layer, batch, logs)
        if on_batch_end is not None:
            self.on_batch_end = lambda batch, logs: on_batch_end(layer, batch, logs)


class SetLayerAttribute(LayerLambdaCallback):
    """ 
    Custom callback for updating a layer's attribute on given training events.
    The value to update the attribute with is given by an optional, anonymous
    function called on the corresponding training event.
    This allows a layer's attribute/hyperparam to be set to a value specific
    for training as a whole or be varied across training epochs or batches.
    """
    def __init__(self,
                 layer: layers.Layer,
                 attr_name: str,
                 on_train_begin: Callable[[], any] = None,
                 on_epoch_begin: Callable[[], any] = None,
                 on_batch_begin: Callable[[], any] = None,
                 verbose: bool=False):
        """
        Initialize the SetLayerAttribute callback so the requested layer's
        attribute will be updated to a value given by one or more anonymous
        functions which are called on training events. 

        :layer: the layer whose attribute we wish to set
        :attr_name: the name of the attribute to set
        :on_train_begin: called to set the value when training begins
        :on_epoch_begin: called to set the value when each epoch begins
        :on_batch_begin: called to set the value when each batch begins
        :verbose: whether the print a message describing the update
        """
        super().__init__(
            layer,
            on_train_begin=lambda lyr, logs: \
                self._set_attr(lyr, attr_name, on_train_begin, verbose) if on_train_begin else None,
            on_epoch_begin=lambda lyr, epoch, logs: \
                self._set_attr(lyr, attr_name, on_epoch_begin, verbose) if on_epoch_begin else None,
            on_batch_begin=lambda lyr, epoch, logs: \
                self._set_attr(lyr, attr_name, on_batch_begin, verbose) if on_batch_begin else None,
        )

    @classmethod
    def _set_attr(cls, layer, attr_name, value_func: Callable[[], any], verbose):
        """ Will set the requested attribute to the value given by value_func """
        if layer and value_func:
            value = value_func()
            setattr(layer, attr_name, value)
            if verbose:
                print(f"{layer.__class__.__name__}({layer.name}) {attr_name} set to {value}")
