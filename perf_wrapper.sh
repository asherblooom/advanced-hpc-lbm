#!/bin/bash
# Check if this is MPI rank 0. On Slurm, SLURM_PROCID holds the rank number.
if [ "$SLURM_PROCID" -eq 0 ]; then
	# Run with perf, outputting to a specific file
	perf stat -e cycles,instructions,cache-references,cache-misses "$@"
else
	# Run normally without perf
	"$@"
fi
