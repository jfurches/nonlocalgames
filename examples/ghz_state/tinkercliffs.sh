#!/bin/bash

#SBATCH --job-name=ghz_state
#SBATCH --account=qc_group
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfurches@vt.edu

#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL

module load site/tinkercliffs/easybuild/setup
module load Anaconda3/2022.10

ENV=~/env/pennylane
source activate $ENV

XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
OMP_NUM_THREAD=1

python try_random_seeds.py -p $SLURM_NTASKS -n 100
