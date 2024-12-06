""" Custom Keras callbacks """
from typing import Union
from datetime import timedelta, datetime
from keras import callbacks

class TrainingTimeoutCallback(callbacks.Callback):
    """ A custom Callback which will stop training after a specified time span. """
    def __init__(self, timeout: Union[timedelta, float]=timedelta(seconds=3600), verbose: bool=0):
        """
        Create a new TrainingTimeoutCallback. This will set up a timer on model training
        which will stop if not complete after a chosen timespan.

        :timeout: the timespan within which training is expected to complete.
        Either a timedelta or a number which is interpreted as seconds.
        :verbose: whether to output a message when an action is taken
        """
        if isinstance(timeout, (int, float)):
            timeout = timedelta(seconds=timeout)
        self.timeout = timeout
        self.start_time = datetime.now()
        self.verbose = verbose
        super().__init__()

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        if self.verbose:
            print(f"Starting the training time-out timer at {self.start_time:%Y-%m-%d %H:%M:%S}.",
                f"Training will be stopped if not completed within {self.timeout} (H:M:S).")
        return super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        elapsed = datetime.now() - self.start_time
        if elapsed > self.timeout:
            if self.verbose > 0:
                print(f"Training has been timed out after {elapsed} (H:M:S) has elapsed.")
            self.model.stop_training = True
        return super().on_epoch_end(epoch, logs)
