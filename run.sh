#!/bin/bash

# Hides any CUDA devices so we get repeatable results
export CUDA_VISIBLE_DEVICES=-1

# Used within the python modules to locate the jktebop executable
# Latest jktebop source and setup instructions at
#   https://www.astro.keele.ac.uk/jkt/codes/jktebop.html
export JKTEBOP_DIR=~/jktebop/

# Make sure the MIST iso files used to create the synthetic-mist-tess-ds are in place
mist_dir=./traininglib/data/mist
isos_stem=MIST_v1.2_vvcrit0.4_basic_isos
if [ ! -d ${mist_dir}/${isos_stem} ]; then
    echo "Creating directory for MIST iso files -- ${mist_dir}/${isos_stem}"
    mkdir -p ${mist_dir}/${isos_stem}
fi

if ls ${mist_dir}/${isos_stem}/*.iso 1> /dev/null 2>&1 ; then
    echo "Found existing MIST iso files"
else
    echo "MIST iso files not found. Will download and extract them"
    wget -P ${mist_dir} http://waps.cfa.harvard.edu/MIST/data//tarballs_v1.2/${isos_stem}.txz
    tar -xvf ${mist_dir}/${isos_stem}.txz -C ${mist_dir}
fi

# Setup/activate the .ebop_maven venv
if [ ! -d ".ebop_maven" ]; then
    echo "Creating the .ebop_maven venv"
    python -m venv .ebop_maven
    source .ebop_maven/bin/activate
    pip install -r requirements.txt
    ipython kernel install --user --name=.ebop_maven
else
    echo "Activating existing .ebop_maven venv"
    source .ebop_maven/bin/activate
fi

# Create the testing and training datasets
python make_formal_test_dataset.py
python make_synthetic_test_dataset.py
python make_training_dataset.py

# Train the ebop maven model
python make_trained_cnn_model.py

# Test the newly trained model
python model_testing.py ./drop/training/cnn-new-ext0-bp-4096-none-500k/*.keras