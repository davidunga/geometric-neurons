#!/bin/zsh
conda_env=geometric-neurons
project_root=~/geometric-neurons
py_file=geometric_encoding/cv_trian_embedding.py

source ~/miniconda3/bin/activate
conda activate $conda_env
cd $project_root
export PYTHONPATH='.'
python $py_file