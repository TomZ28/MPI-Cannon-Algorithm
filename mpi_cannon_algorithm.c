#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

/**
 * Perform parallelized matrix multiplication with MPI.
 * 
 * Usage:
 * 		$ mpirun -np 4 ./mpi_cannon_algorithm <matrix_file_A> <matrix_file_B> <output_file>
 * 
 * The command line arguments are defined as follows:
 * 		- <matrix_file_A> A .txt file containing matrix A
 * 		- <matrix_file_B> A .txt file containing matrix B
 * 		- <output_file> A .txt file to write the result of A*B to
 * 
 * Note that the files containing the matrices should be .txt files containing
 * a square number of integers. For example, 100 is a square number because
 * 100 equals 10 squared.
 * 
 * Example execution:
 * 		$ mpirun -np 4 ./mpi_cannon_algorithm A.txt B.txt C.txt
 */

/* Returns a non-zero number if and only if `n` is a square number
   (e.g. 9 is a square number as 9 is 3 squared), and 0 otherwise. */
int is_square_number(int n) {
	double sqrt_n = sqrt(n);
	return floor(sqrt_n) - sqrt_n == 0;
}

/* Get the size of a matrix from file `fp`. Returns 0 if the matrix is empty,
   not square, or an error occured. */
int get_matrix_size_from_file(FILE *fp) {
	int n;
	int total = 0;
	
	fseek(fp, 0, SEEK_SET); // Go to beginning of file

	while (fscanf(fp, "%d", &n) != EOF) {
		total += 1;
	}

	if (!is_square_number(total)) {
		return 0;
	}

	return (int)sqrt(total);
}

/* Read a matrix of shape (`size` * `size`) from file `fp` into `M`
   (also of the same size) as contiguous memory based on the number
   of processors `n`, where `n` is a perfect square.

   Note that the matrix must be evenly divided into `n` tiles.
   
   Returns 0 if and only if the operation is successful. */
int read_matrix_from_file(int *M, FILE *fp, int size, int n) {
	int sqrt_n = (int)sqrt(n);

    /* Return a non-zero number if `n` is not a perfect square or
       the matrix cannot be evenly divided into `n` tiles. */
	if (!is_square_number(n) || size % sqrt_n != 0) {
		return 1;
	}

    int dim = size / sqrt_n;
	int value;
	fseek(fp, 0, SEEK_SET); // Go to beginning of file

    /* While this loop looks fairly nested, its runtime is simply O(length(M)),
       i.e. linear with respect to the total size of the matrix. */
    for (int i = 0; i < sqrt_n; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < sqrt_n; k++) {
                for (int l = 0; l < dim; l++) {
                    fscanf(fp, "%d", &value);
                    M[i*size*dim + j*dim + k*dim*dim + l] = value;
                }
            }
        }
    }

	return 0;
}

/* Write matrix M of shape (size * size) in contiguous memory form
   to file `fp` in the correct ordering. 
   
   Returns 0 if and only if the operation is successful. */
int write_matrix_to_file(int *M, FILE *fp, int size, int n) {
	int sqrt_n = (int)sqrt(n);

    /* Return a non-zero number if `n` is not a perfect square or
       the matrix cannot be evenly divided into `n` tiles. */
	if (!is_square_number(n) || size % sqrt_n != 0) {
		return 1;
	}

    int dim = size / sqrt_n;
	int count = 0;
	fseek(fp, 0, SEEK_SET); // Go to beginning of file

    /* While this loop looks fairly nested, its runtime is simply O(length(M)),
       i.e. linear with respect to the total size of the matrix. */
    for (int i = 0; i < sqrt_n; i++) {
        for (int j = 0; j < sqrt_n; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
					fprintf(fp, "%d ", M[i*size*dim + j*dim + k*size + l]);
					if (++count % size == 0) {
						fprintf(fp, "\n");
					}
                }
            }
        }
    }

	return 0;
}

/* Given matrices A and B both of shape (size * size), compute and write the 
   result of their product into C. */
void matmul_sequential(int *A, int *B, int *C, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i*size + j] += A[i*size + k] * B[k*size + j];
			}
		}
	}
}

/* Given matrices A and B both of shape (size * size), compute and write the 
   result of their product into C. This version of matmul utilises Cannon's
   algorithm and MPI to perform matrix multiplication in parallel. */
void matmul_MPI(int *A, int *B, int *C, int size, MPI_Comm comm) {
	MPI_Status status;
	MPI_Comm cart_comm;

	int comm_size;
	int dims[2], periods[2];
	int rank, cart_rank, cart_coords[2];
	int up_rank, down_rank, left_rank, right_rank, coords[2];
	int shift_src, shift_dst;

	/* Dimensions for the 2D grid is (sqrt(comm_size) * sqrt(comm_size)),
	   (assuming that comm_size is a square number). */
	MPI_Comm_size(comm, &comm_size);
	dims[0] = dims[1] = sqrt(comm_size);

	// The 2D grid wraps around in both directions (i.e. is periodic).
	periods[0] = periods[1] = 1;

	// Create Cartesian communicator in cart_comm with reordering enabled.
	MPI_Cart_create(comm, 2, dims, periods, 1, &cart_comm);

	/* Get this process's rank and coordinates in the Cartesian topology,
	   in addition to the ranks of the vertical & horizontal shifts. */
	MPI_Comm_rank(cart_comm, &cart_rank);
	MPI_Cart_coords(cart_comm, cart_rank, 2, cart_coords);
	MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank); 
	MPI_Cart_shift(cart_comm, 0, 1, &up_rank, &down_rank); 

	// Set up the initial configuration/skew for Cannon's algorithm.
	MPI_Cart_shift(cart_comm, 1, -cart_coords[0], &shift_src, &shift_dst);
	MPI_Sendrecv_replace(A, size*size, MPI_INT, shift_dst, 1, shift_src, 1, cart_comm, &status);
	MPI_Cart_shift(cart_comm, 0, -cart_coords[1], &shift_src, &shift_dst);
	MPI_Sendrecv_replace(B, size*size, MPI_INT, shift_dst, 1, shift_src, 1, cart_comm, &status);

	// For each step, multiply a pair of chunks, then shift.
	for(int i = 0; i < dims[0]; i++) {
		// Multiply chunks of A and B.
		matmul_sequential(A, B, C, size);

		// Shift A left, and B up.
		MPI_Sendrecv_replace(A, size*size, MPI_INT, left_rank, 1, right_rank, 1, cart_comm, &status);
		MPI_Sendrecv_replace(B, size*size, MPI_INT, up_rank, 1, down_rank, 1, cart_comm, &status);
	}

	// Restore the original layout of A and B.
	MPI_Cart_shift(cart_comm, 1, cart_coords[0], &shift_src, &shift_dst);
	MPI_Sendrecv_replace(A, size*size, MPI_INT, shift_dst, 1, shift_src, 1, cart_comm, &status);
	MPI_Cart_shift(cart_comm, 0, cart_coords[1], &shift_src, &shift_dst);
	MPI_Sendrecv_replace(B, size*size, MPI_INT, shift_dst, 1, shift_src, 1, cart_comm, &status);
	
	MPI_Comm_free(&cart_comm);
}


int main(int argc, char *argv[]) {
	int *A = NULL, *B = NULL, *C = NULL;
	int size_A = 0, size_B = 0, total_size = 0;
	int size[1];
	int items_per_process = 0;
	int *A_chunk, *B_chunk, *C_chunk;
	double start, stop;
	int rank, comm_size;
	int n_chunks, dim;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (rank == 0) {
		// Abort if the number of command line arguments is incorrect.
		if(argc != 4) {
			printf("Usage: %s <matrix_file_A> <matrix_file_B> <output_file>\n", argv[0]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Abort if the number of processors is not a square number.
		if (!is_square_number(comm_size)) {
			printf("Error: Number of processors (%d) is not square\n", comm_size);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Record the amount of time it takes to retrieve A and B.
		start = MPI_Wtime();

		// Check the size of A.
		FILE *file_A = fopen(argv[1], "r");

		if (file_A != NULL) {
			size_A = get_matrix_size_from_file(file_A);
		} else {
			printf("Error: Could not open %s\n", argv[1]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		if (size_A == 0) {
			printf("Error: Could not read a square matrix from %s\n", argv[1]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Check the size of B.
		FILE *file_B = fopen(argv[2], "r");

		if (file_B != NULL) {
			size_B = get_matrix_size_from_file(file_B);
		} else {
			printf("Error: Could not open %s\n", argv[2]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		if (size_B == 0) {
			printf("Error: Could not read a square matrix from %s\n", argv[2]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Abort if A and B are not the same size.
		if (size_A != size_B) {
			printf("Error: A and B are not the same size (A=%d, B=%d)\n", size_A, size_B);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		size[0] = size_A;

		/* Abort if the total size of the matrix cannot be equally divided into
		   `comm_size` squares. */
		if (size_A % (int)sqrt(comm_size) != 0) {
			printf("Error: A and B cannot be equally divided into %d squares\n", comm_size);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		A = malloc(size_A * size_A * sizeof(int));
		if (A == NULL || read_matrix_from_file(A, file_A, size_A, comm_size) != 0){
			printf("Error: An error occured when reading from A.txt\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		B = malloc(size_B * size_B * sizeof(int));
		if (B == NULL || read_matrix_from_file(B, file_B, size_B, comm_size) != 0){
			printf("Error: An error occured when reading from B.txt\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Allocate space for the result C.
		C = malloc(size_A * size_A * sizeof(int));
		if (C == NULL){
			printf("Error: Could not allocate space for result\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		fclose(file_A);
		fclose(file_B);

		stop = MPI_Wtime();
		printf("Retrieved A and B in %lf seconds\n", stop - start);
	}

	// Broadcast the original matrix size to all processes.
	MPI_Bcast(size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	total_size = size[0] * size[0];
	items_per_process = total_size / comm_size;

	// Scatter A and B to all processes.
	A_chunk = malloc(items_per_process * sizeof(int));
	MPI_Scatter(A, items_per_process, MPI_INT, A_chunk, items_per_process, MPI_INT, 0, MPI_COMM_WORLD);
	B_chunk = malloc(items_per_process * sizeof(int));
	MPI_Scatter(B, items_per_process, MPI_INT, B_chunk, items_per_process, MPI_INT, 0, MPI_COMM_WORLD);

	// Initialize the result array.
	C_chunk = calloc(items_per_process, sizeof(int));

	// Time the matrix multiplication.
	if (rank == 0) {
		start = MPI_Wtime();
	}

	matmul_MPI(A_chunk, B_chunk, C_chunk, size[0] / (int)sqrt(comm_size), MPI_COMM_WORLD);

	// Print the amount of time it took in total to compute A*B.
	if (rank == 0) {
		stop = MPI_Wtime();
		printf("Completed matrix multiplication in %lf seconds\n", stop - start);
	}

	MPI_Gather(C_chunk, items_per_process, MPI_INT, C, items_per_process, MPI_INT, 0, MPI_COMM_WORLD);

	// The root writes the output of A*B to a file `C.txt`.
	if (rank == 0) {
		start = MPI_Wtime();

		FILE *file_OUTPUT = fopen(argv[3], "w");
		if (file_OUTPUT == NULL) {
			printf("Error: Could not open %s\n", argv[3]);
		} else {
			if (write_matrix_to_file(C, file_OUTPUT, size[0], comm_size) != 0) {
				printf("Error: Could not write output to %s\n", argv[3]);
			}
			fclose(file_OUTPUT);

			stop = MPI_Wtime();
			printf("Exported result to %s in %lf seconds\n", argv[3], stop - start);
		}

		// Free the full matrices.
		free(A);
		free(B);
		free(C);
	}

	// Perform cleanup.
	free(A_chunk);
	free(B_chunk);
	free(C_chunk);

	MPI_Finalize();
	return 0;
}
