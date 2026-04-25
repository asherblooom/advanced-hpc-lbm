#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=OUTPUT.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --exclusive

make clean
make all

echo "1024x1024"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ../d2q9-bgk-inputs/input_1024x1024.params ../d2q9-bgk-inputs/obstacles_1024x1024.dat
echo "2048x2048"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ../d2q9-bgk-inputs/input_2048x2048.params ../d2q9-bgk-inputs/obstacles_2048x2048.dat
echo "4096x4096"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ../d2q9-bgk-inputs/input_4096x4096.params ../d2q9-bgk-inputs/obstacles_4096x4096.dat
