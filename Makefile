CXX = /opt/homebrew/bin/g++-16
MPI_CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall -Wextra

all: kmeans

kmeans: src/kmeans_mpi_omp.cpp
	OMPI_CXX=$(CXX) $(MPI_CXX) $(CXXFLAGS) -o $@ $<

check: kmeans
	OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans --n 4001 --d 5 --k 4 --iterations 5 --validate
	OMP_NUM_THREADS=1 mpirun -np 4 ./kmeans --n 17 --d 1 --k 1 --iterations 3 --validate

benchmark: kmeans
	python3 benchmark.py

.PHONY: all check benchmark
