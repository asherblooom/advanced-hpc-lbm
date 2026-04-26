#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=OUTPUT.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --exclusive

module load PrgEnv-cray
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

make clean
make all

echo "1024x1024"
srun --mpi=cray_shasta ./perf_wrapper.sh ./d2q9-bgk ../d2q9-bgk-inputs/input_1024x1024.params ../d2q9-bgk-inputs/obstacles_1024x1024.dat
echo "2048x2048"
srun --mpi=cray_shasta ./perf_wrapper.sh ./d2q9-bgk ../d2q9-bgk-inputs/input_2048x2048.params ../d2q9-bgk-inputs/obstacles_2048x2048.dat
echo "4096x4096"
srun --mpi=cray_shasta ./perf_wrapper.sh ./d2q9-bgk ../d2q9-bgk-inputs/input_4096x4096.params ../d2q9-bgk-inputs/obstacles_4096x4096.dat
