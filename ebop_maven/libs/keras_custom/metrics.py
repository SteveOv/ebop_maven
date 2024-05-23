""" Custom Keras metrics """
from keras.src.metrics import reduction_metrics
from keras.src.losses import losses

class MeanMetricForLabel(reduction_metrics.MeanMetricWrapper):
    """
    Base class for mean metrics that are reported on a single label
    """
    def __init__(self, fn, label_ix: int, name: str=None, dtype=None):
        self.label_ix = label_ix
        super().__init__(fn=fn, name=name, dtype=dtype)

        # Metric should be minimized during optimization.
        self._direction = "down"

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[:, self.label_ix],
                             y_pred[:, self.label_ix],
                             sample_weight)

    def get_config(self):
        return {"label_ix": self.label_ix, "name": self.name, "dtype": self.dtype}


class MeanAbsoluteErrorForLabel(MeanMetricForLabel):
    """
    A custom metric which allows the reporting of the MAE of a single label
    """
    def __init__(self, label_ix: int, label_name: str=None, name: str=None, dtype=None):
        """
        A custom metric which allows the reporting of the MAE of a single label

        :label_ix: the index of the chosen label (on the 2nd axis)
        :label_name: optional name of the label (defaults to label_ix)
        :name: optional name of the metric (defaults to 'mae[label_name]')
        :dtype: optional data type of the metric
        """
        if not label_name:
            label_name = f"{label_ix}"
        if not name:
            name = f"mae[{label_name}]"
        super().__init__(losses.mean_absolute_error, label_ix, name, dtype)


class MeanSquaredErrorForLabel(MeanMetricForLabel):
    """
    A custom metric which allows the reporting of the MSE of a single label
    """
    def __init__(self, label_ix: int, label_name: str=None, name: str=None, dtype=None):
        """
        A custom metric which allows the reporting of the MSE of a single label

        :label_ix: the index of the chosen label (on the 2nd axis)
        :label_name: optional name of the label (defaults to label_ix)
        :name: optional name of the metric (defaults to 'mse[label_name]')
        :dtype: optional data type of the metric
        """
        if not label_name:
            label_name = f"{label_ix}"
        if not name:
            name = f"mse[{label_name}]"
        super().__init__(losses.mean_squared_error, label_ix, name, dtype)
