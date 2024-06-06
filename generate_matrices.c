#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Generates two matrices A and B of size (matrix_size * matrix_size), and 
   then writes them to respective files `A.txt` and `B.txt`. The values in each
   matrix will be between 0 and 99. */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
		exit(1);
    }

    int n = atoi(argv[1]);

    FILE *file_A = fopen("A.txt", "w+");
    FILE *file_B = fopen("B.txt", "w+");

    for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
            if (j < n-1) {
                fprintf(file_A, "%d ", rand() % 100);
                fprintf(file_B, "%d ", rand() % 100);
            } else {
                fprintf(file_A, "%d", rand() % 100);
                fprintf(file_B, "%d", rand() % 100);
            }
		}
        fprintf(file_A,"\n");
        fprintf(file_B,"\n");
	}

    fclose(file_A);
    fclose(file_B);
}