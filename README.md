# EBOP Model Automatic input Value Estimation Neural network
A machine learning model for predicting eclipsing binary light curve fitting parameters for
formal analysis with [JKTEBOP](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html).

Detailed instructions on setting up the runtime environment, training & testing datasets,
and training a model can be found [here in the wiki](./readme-setup.md)

An article on the development of EBOP MAVEN is being prepared for publication.

## Overview
The EBOP MAVEN is a Convolutional Neural Network (CNN) machine learning regression model
which accepts phase-folded light curves of detached eclipsing binary (dEB) systems as its
input features in order to predict the input parameters for subsequent formal analysis by
[JKTEBOP](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html). The predicted parameters are:

- the sum ($r_A+r_B$) and ratio ($k \equiv r_B/r_A$) of the stars' fractional radii
  - named `rA_plus_rB` and `k`
- the stars' central brightness ratio ($J$)
  - named `J`
- the orbital eccentricity and argument of periastron through the Poincaré elements ($e\cos{\omega}$ and $e\sin{\omega}$)
  - named `ecosw` and `esinw`
- the orbital inclination through the primary impact parameter ($b_P$)
  - named `bP`

CNN models are widely used in computer vision scenarios. They are often used for classification
problems, for example in classifying Sloan Digital Sky Survey (SDSS) DR16 targets as stars, quasars
or galaxies (Chaini et al. 2023), however here we are using one to address a regression problem.

| ![cnn-ext-model](https://github.com/user-attachments/assets/210e53f2-901b-4a9b-b4c3-366c7dc57a40) |
| :-: |
| _Figure 1. The EBOP MAVEN CNN model. Network visualized using a fork of PlotNeuralNet (Iqbal 2018)._  |

A CNN model consists of one or more convolutional layers which during training "learn" convolution
kernels to isolate important features in the input data, followed by a neural network which learns
to make predictions from the detected features. The EBOP MAVEN model is presented in Fig. 1.
It shows the sets of 1-D convolutional layers which extract features from the input data consisting
of a phase-folded and phase-normalized dEB light-curve with the fluxes converted to magnitudes
relative to zero. As the input data is passed on from each convolutional layer its spatial extent
halved, as is also the case with the MaxPooling layers after each pair of convolutional layers
(they do nothing else). This process progressively reduces the spatial extent of the input data.
Conversely the number of filters is steadily increased from 16 to 128 which extends the number
of features the layers output, with these features covering progressively broader sections of the
light curve. The final output from the convolutional layers are two sets of 128 features
which are flattened to a single array of 256 and then passed into a deep neural network
(DNN) which bases its predictions on the features.

Dropout layers are used after each of the two full DNN layers. These randomly deactivate, by
setting to zero, a proportion of the preceding layer's output on each training step. This
is a common approach to combating overfitting of the training data by preventing neurons
becoming overly dependent on all but the strongest few connections with its inputs.

The model is trained with a Nadam optimizer using an exponentially decaying learning rate.
The training loss function used is the mean average error (MAE) which is less affected by large
losses than the often used mean square error (MSE) and consistently gives better results this case.
The activation functions used are the ReLU function for convolutional layers and the LeakyReLU
function for the the DNN layers (which leaks a small value when negative to mitigate the risk of
dead neurons).

Training is based on the formal-training-dataset which is made up of 250,000 fully synthetic
instances split 80:20 between training and validation datasets. During training the dataset
pipeline includes augmentations which randomly add Gaussian noise and a shift to each
instance's mags feature. The augmentations supplement the Dropout layers in mitigating overfitting
and expose the model to less than perfect data during training, improving its performance
with real data.

## Example usage
The easiest way to use the EBOP MAVEN model is via the Estimator class which provides a `predict()`
function for making predictions and numerous attributes to describe the model and its requirements.
```python
from ebop_maven.estimator import Estimator

# Loads the default model
estimator = Estimator()

# Get the expected size and wrap to apply to model's input "mags" feature
mags_bins = estimator.mags_feature_bins         # 4096
wrap_phase = estimator.mags_feature_wrap_phase  # 0.75 (so phase > 0.75 is wrapped by -1)
```

The Jupyter page [model_interactive_tester.ipynb](./model_interactive_tester.ipynb) more fully
demonstrates the use of the Estimator class and other code within `ebop_maven` for interacting
with JKTEBOP and its inputs & ouputs and for analysing light curves, albiet in the context of the
fixed set of curated targets which make up the formal test dataset. In this example we look at
fitting the TESS timeseries photometry for one of these targets, ZZ Boo sector 50 (see Fig. 2).
The reference analysis for this system is taken from Southworth (2023).

| ![ZZ Boo light curve and phase folded mags feature](https://github.com/user-attachments/assets/22c381d5-3449-406b-b4bd-4c9f9b576ce7) |
| :-: |
| _Figure 2. The light curve for ZZ Boo sector 50 where the SAP fluxes have been converted to magnitudes then rectified to zero with the subtraction of a low order polynomial (left) and the equivalent phase-folded and phase-normalized light curve overlaid with the mags feature from which predictions are made (right)._  |

The input feature for the Estimator's `predict()` function is a numpy array of shape (#instances,
#mags_bins). For each instance it expects a row of size `mags_bins` sampled from the phase-folded
magnitudes data and wrapped above `wrap_phase` (Fig 2 right). It will return its predictions as
a numpy structured array of shape (#instances, #parameters) where values can be accessed via their
parameter/label name (as listed in the Estimator's `label_names` attribute).

```python
# Make a prediction on a single instance using the MC Dropout with 1000 iterations.
# include_raw_preds=True makes predict return a tuple including values for each iteration.
inputs = np.array([mags])
predictions, raw_preds = estimator.predict(inputs, iterations=1000, include_raw_preds=True)

# predictions is a structured array[UFloat] & can be accessed with label names. The dtype is
# UFloat from the uncertainties package which publishes nominal_value and std_dev attributes.
# The following gets the nominal value of k for the first instance.
k_value = predictions[0]["k"].nominal_value
```
The Estimator can make use of the MC Dropout algorithm (Gal & Gharhamani 2016) in order to provide
predictions with uncertainties. Simply set the `predict(iterations)` argument to a value >1 and the
Estimator will make the requested number of predictions on each instance, with the model's Dropout
layers enabled. In this configuration predictions are made for each iteration with a random subset
of the neural network's neurons disabled, with the final predictions returned being the mean and
standard deviation over every iteration for each instance. With dropout enabled the prediction
for each iteration is effectively made with a weak predictor, however given sufficient iterations
the resulting probability distribution represents a strong prediction through the wisdom of crowds.

| ![ZZ Boo violin plot](https://github.com/user-attachments/assets/cc4ef9c7-9221-4881-abda-77aa514ad7d1) |
| :-: |
| _Figure 3. A violin plot of the full set of MC Dropout predictions for ZZ Boo with the horizontal bars showing the mean and standard deviation for each prediction._ |

The final set of prediction nominal values and the label values used for testing are shown below.
The model does not predict $inc$ directly so it has to be calculated from the other predicted values:
```text
---------------------------------------------------------------------------------------------------
ZZ Boo | rA_plus_rB         k         J     ecosw     esinw        bP       inc       MAE       MSE
---------------------------------------------------------------------------------------------------
Label  |   0.236690  1.069100  0.980030  0.000000  0.000000  0.208100 88.636100
Pred   |   0.236385  1.067168  1.006427  0.000909  0.007463  0.272921 88.198103
O-C    |   0.000305  0.001932 -0.026397 -0.000909 -0.007463 -0.064821  0.437997  0.077118  0.028114
===================================================================================================
MAE    |   0.000305  0.001932  0.026397  0.000909  0.007463  0.064821  0.437997  0.077118
MSE    |   0.000000  0.000004  0.000697  0.000001  0.000056  0.004202  0.191842            0.028114
```
 
The predicted values for $r_A+r_B$, $k$, $J$, $e\cos{\omega}$ and $e\sin{\omega}$ and the derived
value for $inc$ can now be used as input parameters for analysis with JKTEBOP. The following shows
the results of analysing the ZZ Boo sector 50 light curve data with task 3, which finds the best
fit to the observations with formal error bars. The fitted params are written to a .par file,
which we can parse to get the values of the parameters of interest.  Shown below is the result
of fitting the parameters previously predicted and how they compare to the labels derived from
the reference analysis:
```
---------------------------------------------------------------------------------------------------
ZZ Boo | rA_plus_rB         k         J     ecosw     esinw        bP       inc       MAE       MSE
---------------------------------------------------------------------------------------------------
Label  |   0.236690  1.069100  0.980030  0.000000  0.000000  0.208100 88.636100
Fitted |   0.236666  1.069237  0.978183 -0.000003  0.000061  0.207551 88.639682
O-C    |   0.000024 -0.000137  0.001847  0.000003 -0.000061  0.000549 -0.003582  0.000886  0.000002
===================================================================================================
MAE    |   0.000024  0.000137  0.001847  0.000003  0.000061  0.000549  0.003582  0.000886
MSE    |   0.000000  0.000000  0.000003  0.000000  0.000000  0.000000  0.000013            0.000002
```

The result of the task 3 analysis can be plotted by parsing the .out file written, which contains
columns with the phase, fitted model and residual values (Fig. 4). 

| ![ZZ Boo fit and residuals](https://github.com/user-attachments/assets/03cdcf0e-89a6-48d3-8caa-38307c8d1dd6) |
| :-: |
| _Figure 4. The fitted model and residuals from the JKTEBOP task 3 fitting of ZZ Boo TESS sector 50 based on the predicted input parameters._ |


## References
Chaini S., Bagul A., Deshpande A., Gondkar R., Sharma K., Vivek M., Kembhavi A., 2023, [MNRAS](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.3123C), 518, 3123

Iqbal H., 2018, HarisIqbal88/PlotNeuralNetv1.0.0 (v1.0.0), [Zenodo](https://doi.org/10.5281/zenodo.2526396)

Southworth J., 2023, [The Observatory](https://ui.adsabs.harvard.edu/abs/2023Obs...143...19S), 143, 19

Gal Y., Ghahramani Z., 2016, [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://doi.org/10.48550/arXiv.1506.02142), doi:10.48550/arXiv.1506.02142
