## Setup of runtime environment
This code base was developed within the context of an Anaconda 3 conda environment named
**ebop_maven**. This environment supports Python 3.9+, TensorFlow, Keras, lightkurve, astropy
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
#### JKTEBOP
These codes have a dependency on the JKTEBOP tool for generating and fitting lightcurves. The
installation media and build instructions can be found
[here](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html). The `JKTEBOP_DIR` environment
variable is used by ebop_maven to be locate the executable at runtime and is set to `~/jktebop/`
in the ebop_maven conda env. This may require updating to match the location where JKTEBOP has
been set up.

#### Alternative, venv setup
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
You may need to install the jupyter kernel in the new venv:
```sh
$ ipython kernel install --user --name=.ebop_maven
```
#### The ebop_maven package
Finally there is support for installing the latest (main) version of ebop_maven as a
pip package, however this is still very much "work in progress" and subject to change.
Simply run:
```sh
$ pip install git+https://github.com/SteveOv/ebop_maven
```
This will install the Estimator class, a pre-built default model and the required support
libraries (basically anything within the ebop_maven subdirectory). This supports the use
of the Estimator with the default model within a client application. The code used in the
following steps for training and testing models is not installed.

## Generation of training and testing datasets

#### The formal training dataset
To generate the datasets which will be used to train and test the machine learning model, first run
```sh
$ python3 make_training_dataset.py
```
to generate the the **formal-training-dataset-250k**. This is a synthetic training dataset built
by randomly sampling distributions of JKTEBOP model parameters across its entire parameter space.
It generates 250,000 instances split 80:20 between training and validation sets.

#### The synthetic MIST/TESS test dataset
Next run
```sh
$ python3 make_synthetic_test_dataset.py
```
to build the **synthetic-mist-tess-dataset**. This is the full dataset of synthetic light-curves
generated from physically plausible systems based on MIST stellar models and the TESS photometric
bandpass. It generates 20,000 randomly oriented instances based on an initial random selection
of metallicity, age and initial masses supplemented with lookups of stellar parameters in the
isochrones.

This module depends on
[MIST isochrone files](http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_basic_isos.txz)
which are not distributed as part of this GitHub repo. You will need to download and extract a
pre-built model grid by following the instructions in
[readme.txt](./traininglib/data/mist/MIST_v1.2_vvcrit0.4_basic_isos/readme.txt). 

#### The formal test dataset of real systems
Finally run
```sh
$ python3 make_formal_test_dataset.py
```
to build the **formal-test-dataset**. These are set of real, well characterized systems from
[DEBCAT](https://www.astro.keele.ac.uk/jkt/debcat/) selected on the availability of _TESS_
lightcurves, suitability for fitting with JKTEBOP and a published characterization from which
parameters can be taken. The chosen systems are configured in the file
`./config/formal-test-dataset.json` which contains the search criteria, labels and supplementary
information for each. 

These steps will take roughly one to two hours on a moderately powerful system, with the resulting datasets taking up ~10 GB of disk space under the `./datasets/` directory.

## Training and testing the machine learning model
The default machine learning model can be built and tested by running the following:
```sh
$ python3 make_trained_cnn_model.py
```
This will create the default CNN/DNN model, trained and validated on the
**formal-training-dataset** to predict the $r_A+r_B$, $k$, $J$, $e\cos{\omega}$,
$e\sin{\omega}$ and $b_P$ labels. Once trained it is evaluated on the
**synthetic-mist-tess-dataset** before a final evaluation on the real systems of
the **formal-test-dataset**.

> By default CUDA cores are disabled so that training and testing is repeatable. 
> This is achieved by setting CUDA_VISIBLE_DEVICES to -1 in environment.yaml used to create
> the ebop_maven conda environment. In this configuration the process above takes about
> one and a half hours on my laptop with an 8 core 11th gen Intel i7 CPU. If you wish to use
> CUDA cores then you will need to change this setting.
>
> Note: there are recorded incidents where TensorFlow v2.16.1 does not "see" installed GPUs
> (me, for one) and under these circumstances the above change may have no effect. This issue
> appears to be resolved with v2.17 (however this version leads to changed training result).

The compiled and trained model will be saved to the 
`./drop/training/cnn-new-ext0-4096-0.75-250k/default-model.keras` file.
Plots of the learning curves and the model structure are written to the `plots` sub-directory.

A detailed evaluation of any models can be invoked with the following command:
```sh
$ python3 model_testing.py [model_files ...]
```

This will initially evaluate model predictions against the **synthetic-mist-tess-dataset**
and the **formal-test-dataset**. Subsequently it will run the full end-to-end testing of
model predictions and JKTEBOP fitting against the **formal-test-dataset**. Testing output
files and a log file will be written to a `testing` sub-directory alongside any tested models. 

You can test the pre-built model, at `./ebop_maven/data/estimator/default-model.keras`, by
running model_testing without any arguments. In this case, the results will be written to
the `./drop/training/published/testing/` directory.

> [!WARNING]  
> The model structure and hyperparameters are still subject to change as ongoing testing and
> model searches continue to reveal improvements.

#### Interactive model tester
This is a jupyter notebook which can be used to download, predict and fit any target
system selected from the **formal-test-dataset** against the pre-built default-model within
the `./ebop_maven/data/estimator` directory or any model found within `./drop/training/`.
It can be run with:
```sh
$ jupyter notebook model_interactive_tester.ipynb
```
