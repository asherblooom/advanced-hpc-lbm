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

echo "128x128"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ./input_128x128.params ./obstacles_128x128.dat
echo "128x256"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ./input_128x256.params ./obstacles_128x256.dat
echo "256x256"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ./input_256x256.params ./obstacles_256x256.dat
echo "1024x1024"
perf stat -e cycles,instructions,cache-references,cache-misses ./d2q9-bgk ./input_1024x1024.params ./obstacles_1024x1024.dat
