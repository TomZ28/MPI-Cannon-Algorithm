# MPI-Cannon-Algorithm
An implementation of [Cannon's algorithm](https://en.wikipedia.org/wiki/Cannon%27s_algorithm) using MPI in C. It is a distributed algorithm for matrix multiplication.

## Dependencies
This project requires an implementation of MPI to be installed to run. For example, [OpenMPI](https://www.open-mpi.org/) and [MPICH](https://www.mpich.org/) are some well-known implementations.

## Compiling the Project
Simply run the following on your command line terminal:
```
make
```

## Usage
```
$ mpirun -np <num_processes> ./mpi_cannon_algorithm <matrix_file_A> <matrix_file_B> <output_file>
```

The command line arguments are defined as follows:
  - `<num_processes>` The number of processes to initialize MPI with
  - `<matrix_file_A>` A .txt file containing matrix A
  - `<matrix_file_B>` A .txt file containing matrix B
  - `<output_file>` A .txt file to write the result of A*B to

### Example execution
```
$ mpirun -np 4 ./mpi_cannon_algorithm A.txt B.txt C.txt
```

## Notes
### Generating Matrices
You can generate the matrices yourself, but if you would like to test the program out, some matrix-generating code is included in this project. After compiling the project, run the generator:
```
./generate_matrices <matrix_size>
```
where `<matrix_size>` is the size length of the matrix.
