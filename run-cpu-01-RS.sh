module=Python/3.10.4-GCCcore-11.3.0
pyfile=analysis/cv_train.py

#BSUB -q new-long
#BSUB -J geometric-neurons
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R "rusage[mem=2048]"
#BSUB -n 2
#BSUB -W 96:00

module load $module
cd ~/geometric-neurons
source bin/activate
git pull
export PYTHONPATH='.'
python $pyfile -n RS
