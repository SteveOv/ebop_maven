# EBOP Model Automatic input Value Estimation Neural network
A machine learning model for estimating input values for characterization of detached eclipsing binaries stars by [JKTEBOP](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html).

Included in this repository are codes for generating training & test datasets, for building, training & testing the machine learning model and _some other stuff still to be decided_.

> [!WARNING]  
> This is a work in progress. Handle with care.

## Installation
This code base was developed within the context of an Anaconda 3 conda environment named ebop_maven. This environment supports Python 3.9+, TensorFlow, Keras, lightkurve, astropy and any further libraries upon which the code is dependent. To set up the ebop_maven conda environment, having first cloned this GitHub repo, open a Terminal, navigate to this local directory and run the following command;
```sh
$ conda env create -f environment.yaml
```
You will need to activate the environment whenever you wish to run any of these modules. Use the following command;
```sh
$ conda activate ebop_maven
```
#### JKTEBOP
These codes have a dependency on the JKTEBOP tool for generating and fitting lightcurves. The installation media and build instructions can be found [here](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html). The `JKTEBOP_DIR` environment variable, used by ebop_maven to be locate the executable at runtime, is set to `~/jktebop/` in the ebop_maven conda env. This will need to match the location where JKTEBOP has been set up.

#### Alternative, venv installation
If you prefer not to use a conda environment, the following venv setup works although I haven't tested it as thoroughly. Again, from this directory run the following to create and activate a .ebop_maven env;
```sh
$ python -m venv .ebop_maven

$ source .ebop_maven/bin/activate
```
Then to set up the required packages in the environment run:
```sh
$ pip install -r requirements.txt
```
#### The ebop_maven package
Finally there is support for installing ebop_maven into other environments as a pip package, however this is still very much "subject to change/failure".  Simply run:
```sh
$ pip install git+https://github.com/SteveOv/ebop_maven
```
## Usage

#### Generation of training and testing datasets
The first step is to generate the datasets which will be used to train and test the machine learning model. These are built by running the following:
```sh
$ python make_training_datasets.py
```
This module will write three datasets under the ./datasets directory:
- **formal-training-dataset** : a synthetic dataset built by randomly sampling distributions of JKTEBOP model parameters
    - currently this generates 100,000 instances split on the ratios 80:20 between training and validation sets
    - the base parameters sampled are $r_A+r_B$, $k$, $i$, $J$, $q_{phot}$, $e$ and $\omega$
- **synthetic-mist-tess-dataset** : a synthetic dataset built from a grid of physical parameters
    - the file `./config/synthetic-mist-tess-dataset.json` gives the grid of base physical parameters; $M_{init}$, $P$, $i$, $e$, $\omega$
    - metallicity is fixed at 0.143 and system age is calculated to be mid to late M-S of the more massive component
    - MIST stellar models are used to fully characterise each instance sufficient to generate synthetic lightcurves with JKTEBOP
- **formal-test-dataset** : a set of real, well characterized systems from [DEBCAT](https://www.astro.keele.ac.uk/jkt/debcat/)
    - selection criteria being the availability of _TESS_ lightcurves, suitability for fitting with JKTEBOP and a published characterization from which parameters can be taken
    - the file `./config/formal-test-dataset.json` contains the search criteria and labels for each target

#### Training and testing the machine learning model
The default machine learning model can be built and tested by running the following:
```sh
$ python make_trained_cnn_model.py
```
This will create the default two input CNN/DNN model. The Mags-Input accepts a (1024, 1) shape
input containing the relative magnitude values of the target's phase folded, light-curve
and the Ext-Input accepts a (2, 1) shape input containing the phase of the secondary eclipse
($\phi_S$) and ratio of the eclipse widths ($d_S/d_P$).

The model will be trained on the **formal-training-dataset** (also the source of the validation
dataset) to predict the $r_A+r_B$, $k$, $J$, $e\,\cos{\omega}$, $e\,\sin{\omega}$ and $i$ labels.
Once trained it is evaluated on the **synthetic-mist-tess-dataset** before a final evaluation
on the real systems of the **formal-test-dataset**.

> By default CUDA cores are disabled so that training and testing is repeatable. In this 
> configuration the process above takes about 10 to 20 minutes on my laptop with an 11th gen
> Intel i7 CPU. If you have them, CUDA cores can be enabled by setting the `ENFORCE_REPEATABILITY`
> const to False to give a significant reduction in training time.
>
> Note: there are recorded incidents where TensorFlow v2.16.1 does not "see" installed GPUs
> (me for one) and under these circumstances the above change may have no effect.

The compiled and trained model will be saved to `./drop/cnn_ext_model.keras`.

Plots of the learning curves and the model structure are written to the `.drop/plots/` directory.

The results of the evaluation with the **formal-test-dataset** are written to the
`./drop/results/{model_name}/{training_set_name}/mags_1024_0.75/` directory. This includes
a csv file which give the MAE and MSE by label, instance & in total, and eps plots
of the predictions vs label values.

> [!WARNING]  
> The model structure and hyperparameters are still subject to change as ongoing testing and
> grid searches continue to reveal improvements.

The final evaluation against the **formal-test-dataset** can be re-run against the saved
model with the following:
```sh
$ python model_testing.py
```

#### Model structure and hyperparameter search
A search over a range of model structures and hyperparameter values, using the 
[hyperopt libarary](http://hyperopt.github.io/hyperopt/)'s tpe.suggest algorithm, can be run with
the following command:
```sh
$ python model_search.py
```
> [!WARNING]  
> This will take a long time! As in hours, if not days.
