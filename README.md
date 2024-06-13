# EBOP Model Automatic input Value Estimation Neural network
A machine learning model for estimating input values for characterization of detached eclipsing
binaries stars by [JKTEBOP](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html).

Included in this repository are codes for generating training & test datasets, for building,
training & testing the machine learning model and _some other stuff still to be decided_.

> [!WARNING]  
> This is a work in progress. Handle with care.

## Installation
This code base was developed within the context of an Anaconda 3 conda environment named
ebop_maven. This environment supports Python 3.9+, TensorFlow, Keras, lightkurve, astropy
and any further libraries upon which the code is dependent. To set up the ebop_maven conda
environment, having first cloned this GitHub repo, open a Terminal, navigate to this local
directory and run the following command;
```sh
$ conda env create -f environment.yaml
```
You will need to activate the environment whenever you wish to run any of these modules.
Use the following command;
```sh
$ conda activate ebop_maven
```
##### JKTEBOP
These codes have a dependency on the JKTEBOP tool for generating and fitting lightcurves. The
installation media and build instructions can be found
[here](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html). The `JKTEBOP_DIR` environment
variable is used by ebop_maven to be locate the executable at runtime and is set to `~/jktebop/`
in the ebop_maven conda env. This may require updating to match the location where JKTEBOP has
been set up.

#### Alternative, venv installation
If you prefer not to use a conda environment, the following venv setup works although I haven't
tested it as thoroughly. Again, from this directory run the following to create and activate the
.ebop_maven env;
```sh
$ python -m venv .ebop_maven

$ source .ebop_maven/bin/activate
```
Then run the following to set up the required packages within the environment:
```sh
$ pip install -r requirements.txt
```
#### The ebop_maven package
Finally there is support for installing ebop_maven as a pip package, however this is still very
much "work in progress" and subject to change.  Simply run:
```sh
$ pip install git+https://github.com/SteveOv/ebop_maven
```
This will install the Estimator class, a pre-built default model and the required support
libraries. The code used in the following steps for training and testing models is not installed.

## Usage

#### Generation of training and testing datasets
The first step is to generate the datasets which will be used to train and test the machine
learning model. These are built by running the following:
```sh
$ python3 make_training_datasets.py
```
This module will write three datasets under the ./datasets directory:
- **formal-training-dataset** : a synthetic dataset built by randomly sampling distributions
of JKTEBOP model parameters across its entire parameter space
    - currently this generates 250,000 instances split on the ratios 80:20 between training
        and validation sets
- **synthetic-mist-tess-dataset** : a synthetic dataset of light-curves from physically plausible
        systems based on MIST stellar models and the TESS photometric bandpass
    - this depends on [MIST isochrone files](http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_basic_isos.txz)
        being downloaded and extracted into the./ebop_maven/libs/data/stellar_models/mist/MIST_v1.2_vvcrit0.4_basic_isos/ directory
    - 20,000 randomly oriented instances will be generated based on an initial random selection
        of metallicity, age and initial masses supplemented with lookups of stellar parameters
        in the isochrones
- **formal-test-dataset** : a set of real, well characterized systems from
        [DEBCAT](https://www.astro.keele.ac.uk/jkt/debcat/)
    - selection criteria being the availability of _TESS_ lightcurves, suitability for fitting
        with JKTEBOP and a published characterization from which parameters can be taken
    - the file `./config/formal-test-dataset.json` contains the search criteria, labels and
        supplementary information for each target

This will roughly hour or so on a moderately powerful system, with the resulting datasets taking up ~15 GB of disk space.

#### Training and testing the machine learning model
The default machine learning model can be built and tested by running the following:
```sh
$ python3 make_trained_cnn_model.py
```
This will create the default CNN/DNN model, trained and validated on the
**formal-training-dataset** to predict the $r_A+r_B$, $k$, $J$, $e\cos{\omega}$,
$e\sin{\omega}$ and $b_P$ labels. Once trained it is evaluated on the
**synthetic-mist-tess-dataset** before a final evaluation on the real systems of
the **formal-test-dataset**.

> By default CUDA cores are disabled so that training and testing is repeatable. In this 
> configuration the process above takes about 10 to 20 minutes on my laptop with an 11th gen
> Intel i7 CPU. If you have them, CUDA cores can be enabled by setting the `ENFORCE_REPEATABILITY`
> const to False to give a significant reduction in training time.
>
> Note: there are recorded incidents where TensorFlow v2.16.1 does not "see" installed GPUs
> (me for one) and under these circumstances the above change may have no effect.

The compiled and trained model will be saved to the 
`./drop/training/cnn-new-ext0-4096-0.75-250k/default-model.keras` file.
Plots of the learning curves and the model structure are written to the `plots` sub-directory.

A detailed evaluation of the newly created model can be invoked with the following command:
```sh
$ python3 model_testing.py ./drop/training/cnn-new-ext0-4096-0.75-250k/default-model.keras
```
Alternatively, you can evaluate the pre-built model within the ./ebop_maven/data/estimator/
directory with
```sh
$ python3 model_testing.py
```

This will initially evaluate model predictions against the **synthetic-mist-tess-dataset**
and the **formal-test-dataset**. Subsequently it will run the full end-to-end testing of
model predictions and JKTEBOP fitting against the **formal-test-dataset**. Testing output
files and a log file will be written to the `testing` sub-directory. 

> [!WARNING]  
> The model structure and hyperparameters are still subject to change as ongoing testing and
> model searches continue to reveal improvements.



#### Model structure and hyperparameter search
A search over a range of model structures and hyperparameter values, using the 
[hyperopt libarary](http://hyperopt.github.io/hyperopt/)'s tpe.suggest algorithm, can be run with
the following command:
```sh
$ python3 model_search.py
```
> [!WARNING]  
> This will take a long time! As in hours, if not days.
