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
The first step is to generate the datasets which will be used to train and test the machine learning model. These are built by running the command below:
```sh
$ python make_training_datasets.py
```
This module will write three datasets under the ./datasets directory:
- **formal-training-dataset** : a synthetic dataset built by randomly sampling distributions of JKTEBOP model parameters
    - currently this generates 100,000 instances split on the ratios 80:20 between training and validation sets
    - the base parameters sampled are $r_A+r_B$, $k$, $i$, $J$, $q_{phot}$, $e$ and $\omega$ ($L_3$ under revirew)
- **synthetic-mist-tess-dataset** : a synthetic dataset built from a grid of physical parameters
    - the file ./config/synthetic-mist-tess-dataset.json gives the grid of base physical parameters; $M_{init}$, $P$, $i$, $e$, $\omega$ ($L_3$ under review)
    - metallicity is fixed at 0.143 and system age is calculated to be mid to late M-S of the more massive component
    - MIST stellar models are used to fully characterise each instance sufficient to generate synthetic lightcurves with JKTEBOP
- **formal-test-dataset** : a set of real, well characterized systems from [DEBCAT](https://www.astro.keele.ac.uk/jkt/debcat/)
    - selection criteria being the availability of _TESS_ lightcurves, suitability for fitting with JKTEBOP and a published characterization from which parameters can be taken
    - the file ./config/formal-test-dataset.json contains the search criteria and labels for each target
