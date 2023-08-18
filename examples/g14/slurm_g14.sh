#!/bin/bash

#SBATCH --job-name=nonlocal-games
#SBATCH --account=qc_group
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfurches@vt.edu

#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL


module load site/tinkercliffs/easybuild/setup
module load Anaconda3/2020.11

module list

# in this package we rely on the adapt-gym package
ADAPTGYM=~/adapt-gym
ENV=~/env/adapt-gym
ENV_YML=$ADAPTGYM/env.yml

source activate $ENV

# by running source activate, if the enviroment is not found,
# it returns a non-zero exit code which is stored in $?

if [ $? -eq 0 ]; then
    # conda env exists 
    echo "Updating conda environment $ENV"
    # conda env update -f $ENV_YML --prune
else
    # install conda env
    conda env create -p $ENV -f $ENV_YML
    source activate $ENV
fi

# install adapt gym package, applying updates as necessary
pip install $ADAPTGYM

# install nonlocalgames package
pip install ../../

python run_g14.py --adapt-tol=1e-6 --dpo-tol=1e-6 \
    --num-cpus=16 \
    --seeds=../../data/seeds.txt \
    --trials=500 \
    --weighting=balanced
