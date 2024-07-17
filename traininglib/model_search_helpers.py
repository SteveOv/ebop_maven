""" Helper functions for model_search """
from typing import Callable, Union, List, Dict, Tuple

import keras
from keras import layers

from ebop_maven import modelling

# pylint: disable=too-many-arguments


def get_trial_value(trial_dict: dict,
                    key: str,
                    pop_it: bool=False,
                    tuples_to_lists: bool=True) -> Union[Callable, any]:
    """
    Will get the requested value from the trial dictionary. Specifically handles the special
    case where we are getting a function/class with hp.choices over the args by parsing the
    function and its kwargs list and then executing it.

    Example of the model "structure". Calling get_trial_value(trials_pspace, "model") returns the
    model resulting from build_mags_ext_model() being executed with the other key/values as its
    kwargs. This also handles the fact that some of these kwargs are nested functions themselves.

    ```Python
    trials_pspace = {
        "model": {
            "func": modelling.build_mags_ext_model,
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
                    "func": model_search_helpers.cnn_fixed_pairs_with_pooling,
                    "num_pairs": hp.uniformint("free_cnn_fixed_num_pairs", low=2, high=4),
                    ...
                },
            },
            ...
        }
    }
    ```

    :trial_dict: the source trials parameter space dictionary
    :key: the key to look for
    :pop_it: whether the pop or get the key
    :tuples_to_lists: undo hyperopt coercing lists into tuples
    """
    # We want a KeyError if item not found
    value = trial_dict.pop(key) if pop_it else trial_dict.get(key)

    # Workaround for the nasty behaviour in hyperopt where lists get silently converted to tuples
    # (see: https://github.com/hyperopt/hyperopt/issues/526)
    if tuples_to_lists and isinstance(value, Tuple):
        value = list(value)

    # We're looking for the special case: a dict with a func/class item and the rest the kwargs.
    if isinstance(value, dict) and ("func" in value or "class" in value):
        the_callable = value.get("func", value.get("class"))

        # support it being a str (easier to read when reporting)
        if isinstance(the_callable, str):
            the_callable = eval(the_callable) # pylint: disable=eval-used

        # Recurse when reading the kwargs to handle those which are funcs/classes
        kwargs = { k: get_trial_value(value, k, tuples_to_lists=tuples_to_lists)
                                        for k in value if k not in ["func", "class"] }

        # Finally, return the result of executing the callable with its kwargs
        return the_callable(**kwargs)

    # Otherwise, fall back on returning the raw value
    return value


# -----------------------------------------------------------
# Functions to create variously structured model building blocks
# -----------------------------------------------------------
def cnn_with_pooling(num_layers: int=4,
                     filters: Union[int, List[int]]=64,
                     kernel_size: Union[int, List[int]]=8,
                     strides: Union[int, List[int]]=None,
                     strides_fraction: float=0.5,
                     padding: str="same",
                     activation: str="relu",
                     pooling_ixs: Union[int, List[int]]=None,
                     pooling_type: layers.Layer=None,
                     pooling_kwargs: Union[Dict, List[Dict]]=None):
    """
    Prototype of creating a set of CNN layers with optional pooling at given indices.

    Useful for a less structured approach than 2*conv+pooling of the other cnn methods.
    """
    if not isinstance(filters, List):
        filters = [filters] * num_layers
    if not isinstance(kernel_size, List):
        kernel_size = [kernel_size] * num_layers
    if not isinstance(strides, List):
        strides = [strides] * num_layers
    if pooling_kwargs is None:
        pooling_kwargs = { "pool_size": 2, "strides": 2 }
    if pooling_ixs and pooling_type and pooling_kwargs and isinstance(pooling_kwargs, dict):
        num_pools = len(pooling_ixs)
        pooling_kwargs = [pooling_kwargs] * num_pools

    def layers_func(input_tensor: keras.KerasTensor) -> keras.KerasTensor:
        # Expected failure if any list isn't num_layers long
        pooling_ix = 0
        for cnn_ix in range(num_layers):
            if pooling_ixs and pooling_type and cnn_ix+pooling_ix in pooling_ixs:
                input_tensor = pooling_type(name=f"Pool-{pooling_ix+1}",
                                            **pooling_kwargs[pooling_ix])(input_tensor)
                pooling_ix += 1

            if not strides[cnn_ix]:
                strides[cnn_ix] = max(1, int(kernel_size[cnn_ix] * strides_fraction))
            if strides[cnn_ix] > kernel_size[cnn_ix]:
                strides[cnn_ix] = kernel_size[cnn_ix]

            input_tensor = layers.Conv1D(filters=int(filters[cnn_ix]),
                                         kernel_size=int(kernel_size[cnn_ix]),
                                         strides=int(strides[cnn_ix]),
                                         padding=padding,
                                         activation=activation,
                                         name=f"Conv-{cnn_ix+1}")(input_tensor)
        return input_tensor
    return layers_func


def cnn_scaled_pairs_with_pooling(num_pairs: int=2,
                                  filters: int=32,
                                  kernel_size: int=16,
                                  strides: int=None,
                                  strides_fraction: float=0.5,
                                  scaling_multiplier: int=2,
                                  padding: str="same",
                                  activation: str="relu",
                                  pooling_type: layers.Layer=None,
                                  pooling_kwargs: Union[Dict, List[Dict]]=None,
                                  trailing_pool: bool=True):
    """
    Pairs of Conv1d layers where the filters & kernel_size can optionally be
    scaled up/down for each successive pair (by scaling_multiplier). Each pair
    can optionally be followed with a pooling layer. If we are including
    pooling layers the trailing_pool flag dictates whether the pooling layer
    after the final pair of Conv1ds is appended or not.

    strides can be specified either as a fixed value (strides) or as a fraction
    of the kernel (strides_fraction) even as the kernel_size is itself scaled.
    If both set, strides takes precendent.

    The pattern of repeating 2*Conv+1*pool with increasing filters/decreasing
    kernels crops up regularly in known/documented CNNs such as;
    - LeNet-5 (LeCun+98)
    - AlexNet (Krishevsky+12)
    - Shallue & Vanderburg (2018, AJ, 155); especially relevant as based on Kepler LCs
    """
    if pooling_kwargs is None:
        pooling_kwargs = { "pool_size": 2, "strides": 2 }
    if not strides and not strides_fraction:
        strides_fraction = 0.5

    def layer_func(input_tensor: keras.KerasTensor) -> keras.KerasTensor:
        this_filters = int(filters)
        this_kernel_size = int(kernel_size)
        this_strides = int(strides) if strides else max(1, int(this_kernel_size * strides_fraction))

        for ix in range(num_pairs):
            for sub_ix in range(2):
                input_tensor = layers.Conv1D(filters=this_filters,
                                             kernel_size=this_kernel_size,
                                             strides=this_strides,
                                             padding=padding,
                                             activation=activation,
                                             name=f"Conv-{ix+1}-{sub_ix+1}")(input_tensor)

            if pooling_type and (trailing_pool or ix < num_pairs-1):
                input_tensor = pooling_type(name=f"Pool-{ix+1}", **pooling_kwargs)(input_tensor)

            if scaling_multiplier != 1:
                this_filters *= scaling_multiplier
                this_kernel_size = max(1, this_kernel_size // scaling_multiplier)
                if not strides or this_strides > this_kernel_size:
                    this_strides = max(1, int(this_kernel_size * strides_fraction))
        return input_tensor
    return layer_func


def cnn_fixed_pairs_with_pooling(num_pairs: int=2,
                                 filters: int=64,
                                 kernel_size: int=4,
                                 strides: int=None,
                                 strides_fraction: float=0.5,
                                 padding: str="same",
                                 activation: str="relu",
                                 pooling_type: layers.Layer=None,
                                 pooling_kwargs: Union[Dict, List[Dict]]=None,
                                 trailing_pool: bool=True):
    """
    Pairs of Conv1d layers with fixed filters, kernel_size and strided and
    optionally followed with a pooling layer.

    Another repeated 2*conv+1*pool structure but this time the
    filters/kernels/strides remain constant across all the conv1d layers.
    It's a very simple structure with the filters "seeing" an ever larger
    FOV as the spatial extent of the input data is reduced as it proceeds
    through the layers.

    strides can be specified either as an explicit value (strides) or a fraction
    of the kernel (strides_fraction). If both set, strides takes precendent.
    
    Experimentation has shown that variations on this structure appear to offer
    a good baseline level of performance.
    """
    return cnn_scaled_pairs_with_pooling(num_pairs, filters, kernel_size,
                                         strides, strides_fraction, 1,
                                         padding, activation,
                                         pooling_type, pooling_kwargs, trailing_pool)


def dnn_with_taper(num_layers: int,
                   units: int,
                   kernel_initializer: any,
                   activation: any,
                   dropout_rate: float=0,
                   taper_units: int=0) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ 
    Creates the function to build the requested DNN layers
    """
    def layers_func(prev_tensor: keras.KerasTensor) -> keras.KerasTensor:
        prev_tensor = modelling.hidden_layers(int(num_layers), int(units), kernel_initializer,
                                              activation, dropout_rate,
                                              name_prefix=("Hidden-", "Dropout-"))(prev_tensor)
        if taper_units:
            prev_tensor = modelling.hidden_layers(1, int(taper_units), kernel_initializer,
                                                  activation, name_prefix=("Taper-", ))(prev_tensor)
        return prev_tensor
    return layers_func
