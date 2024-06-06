CC=gcc
MPICC?=mpicc

PROGS=generate_matrices mpi_cannon_algorithm

all: ${PROGS}

generate_matrices: generate_matrices.c
	${CC} -o $@ $^

mpi_cannon_algorithm: mpi_cannon_algorithm.c
	${MPICC} -o $@ $^ -lm

clean:
	rm -f ${PROGS}
