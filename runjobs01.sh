name=geometric-neurons
module=Python/3.10.4-GCCcore-11.3.0
pyfile=analysis/cv_train.py

#BSUB -q new-long
#BSUB -J $name
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R "rusage[mem=4096]"
#BSUB -n 8
#BSUB -W 24:00

module load $module
cd ~/$name
source bin/activate
git pull
export PYTHONPATH='.'
python $pyfile

