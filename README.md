# EBOP Model Automatic input Value Estimation Neural network
A machine learning model for estimating input values for characterization of detached eclipsing binaries stars by [JKTEBOP](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html).

Included in this repository are codes for generating training & test datasets, for building, training & testing the machine learning model and _some other stuff still to be decided_.

## Installation
This code base was developed within the context of an Anaconda 3 conda environment named ebop_maven. This environment supports Python 3.9+, TensorFlow, Keras, lightkurve, astropy and any further libraries upon which the code is dependent. To set up the ebop_maven conda environment, having first cloned this GitHub repo, open a Terminal, navigate to this local directory and run the following command;
```sh
$ conda env create -f environment.yaml
```
You will need to activate the environment whenever you wish to run any of these modules. Use the following command;
```sh
$ conda activate ebop_maven
```
#### Alternative installation
If you prefer not to use a conda environment, the following venv setup works although I haven't tested it as thoroughly. Again, from this directory run the following to create and activate a .ebop_maven env;
```sh
$ python -m venv .ebop_maven

$ source .ebop_maven/bin/activate
```
Then to set up the required packages in the environment run:
```sh
$ pip install -r requirements.txt
```

Finally, there is support for installing ebop_maven into other environments as a pip package, however this pre-release functionality and subject to failures and/or change.  Simply run:
```sh
$ pip install git+https://github.com/SteveOv/ebop_maven
```

## Usage
TODO
