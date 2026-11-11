# DSA5208 Project 3: Hybrid MPI/OpenMP K-means

Group members: Zimo Xia, Yixiao Xu

## AI usage disclosure

The code for this project was generated using AI. The project structure was
designed by the group members. The group members reviewed and checked the
AI-generated code and take responsibility for the final submission.

This project implements fixed-iteration Lloyd K-means in C++17. MPI divides the
observations into contiguous blocks across processes. OpenMP divides each
process's observations among threads; thread-private sums and counts avoid
write races. Two `MPI_Allreduce` calls synchronize the sufficient statistics
after each iteration. Empty clusters keep their previous centroid.

Public source repository: https://github.com/FuugaMo/DSA5208-Project3

## Build

The local build uses Open MPI and GCC 16 with OpenMP:

```sh
make
```

On a cluster, override `CXX` if another GNU C++ compiler is installed:

```sh
make CXX=/path/to/g++
```

## Run and validate

```sh
OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans --n 4001 --d 5 --k 4 --iterations 5 --validate
make check
```

The `--validate` option computes an independent serial reference on rank 0
after the timed parallel region. It fails if any final centroid differs by
more than `1e-8`. It is intended for small cases; the serial reference
regenerates all observations and is expensive for large inputs.

Output is one JSON object with the maximum rank elapsed time, maximum rank
collective time, final within-cluster sum of squares (inertia), label
agreement against the synthetic generating clusters, and validation error.
The runtime includes assignment, local accumulation, reductions, and centroid
updates over all iterations. It excludes data generation and final quality
metrics. Because each rank generates its assigned global indices from a
fixed hash, all process counts use precisely the same strong-scaling dataset.

## Reproduce the experiments

```sh
python3 benchmark.py --repetitions 5
```

The script writes individual runs to `results/raw.jsonl`, summaries to
`results/summary.csv`, and the local software/platform record to
`results/machine.json`. Strong scaling holds `N=1,000,000`, `D=32`,
`K=16`, and 15 iterations fixed. Weak scaling holds 250,000 observations
per MPI process, with one thread per process. Each configuration is repeated
five times and summarized by the median. Launch overhead is excluded because
timing uses `MPI_Wtime` inside the executable.

To record objective values at several iteration counts, run
`python3 convergence.py`; it writes `results/convergence.csv`.

## Atlas 8 results

The completed PBS runs are included under `results/hpc-atlas8` (two nodes,
45 measurements) and `results/hpc-atlas8-one-node` (one node, 35 measurements).
Both job logs report exit status 0. The two-node placement record shows three
MPI ranks on each compute node. The report source in `report/` includes the cluster scaling tables and plot
as well as the local results. Compile `report/main.tex` to produce the PDF.

## Files

- `src/kmeans_mpi_omp.cpp`: implementation and serial reference.
- `benchmark.py`: experiment driver.
- `results/`: raw and summarized measurements.
- `report/`: report source and rendered PDF.

The Atlas 8 job in `hpc/atlas8.pbs` reserves two 12-core nodes from the
`parallel` PBS queue, loads GCC 13.2, Open MPI 4.1.6, and Python 3.11,
rebuilds and validates the Linux executable, then runs strong and weak scaling.
It verifies that MPI ranks occupy both nodes and disables MPI core binding so
OpenMP threads can use the CPUs assigned to the PBS job.
The job uses PBS's `pbs_tmrsh` as Open MPI's remote launch agent; Atlas compute
nodes otherwise reject the default SSH launch from this account.
Submit from the project root on Atlas with `qsub hpc/atlas8.pbs`. The job writes
measurements and host-placement records to `results/hpc-atlas8`, leaving local
measurements intact. `hpc/atlas8-one-node.pbs` runs the corresponding 12-CPU
one-node experiment. Regenerate the two-node tables and plot with:

```sh
python3 make_figures.py --summary results/hpc-atlas8/summary.csv \
  --output-dir report/hpc --label-prefix hpc-
```
