module=Python/3.10.4-GCCcore-11.3.0
pyfile=analysis/cv_train.py

#BSUB -q gpu-medium
#BSUB -J geometric-neurons
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R "rusage[mem=4096]"
#BSUB -n 8
#BSUB -W 24:00
#BSUB -gpu "num=1"

module load $module
cd ~/geometric-neurons
source bin/activate
git pull
export PYTHONPATH='.'
python $pyfile
