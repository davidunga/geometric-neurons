#!/bin/zsh
conda_env=geometric-neurons
project_root=~/geometric-neurons
py_file=analysis/data_maker.py

source ~/miniconda3/bin/activate
conda activate $conda_env
cd $project_root
export PYTHONPATH='.'
python $py_file