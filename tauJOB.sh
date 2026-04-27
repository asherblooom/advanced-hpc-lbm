#!/bin/bash
#SBATCH --job-name=tau_profiling
#SBATCH --output=OUTPUT.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --exclusive

module load PrgEnv-cray
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PATH=/home/b35cg/asher.b35cg/tau-2.35.1/craycnl/bin:$PATH
export TAU_MAKEFILE=/home/b35cg/asher.b35cg/tau-2.35.1/craycnl/lib/Makefile.tau-cray-mpi-pdt

make clean
make CC=tau_cc.sh

echo "1024x1024"
mkdir -p profiles_1024
export PROFILEDIR=profiles_1024
srun --mpi=cray_shasta ./d2q9-bgk ../d2q9-bgk-inputs/input_1024x1024.params ../d2q9-bgk-inputs/obstacles_1024x1024.dat

echo "2048x2048"
mkdir -p profiles_2048
export PROFILEDIR=profiles_2048
srun --mpi=cray_shasta ./d2q9-bgk ../d2q9-bgk-inputs/input_2048x2048.params ../d2q9-bgk-inputs/obstacles_2048x2048.dat

echo "4096x4096"
mkdir -p profiles_4096
export PROFILEDIR=profiles_4096
srun --mpi=cray_shasta ./d2q9-bgk ../d2q9-bgk-inputs/input_4096x4096.params ../d2q9-bgk-inputs/obstacles_4096x4096.dat
