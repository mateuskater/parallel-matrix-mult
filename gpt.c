#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stddef.h>
#include "chrono.c"

#define MATRIX_SIZE 4

void print_matrix(int *matrix) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%d\t", matrix[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_multiply(int *matrix_a, int *matrix_b, int *result, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i * MATRIX_SIZE + j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                result[i * MATRIX_SIZE + j] += matrix_a[i * MATRIX_SIZE + k] * matrix_b[k * MATRIX_SIZE + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int world_rank, world_size;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != MATRIX_SIZE) {
        fprintf(stderr, "O número de processos deve ser igual ao tamanho da matriz.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *matrix_a = NULL;
    int *matrix_b = NULL;
    int *submatrix_a = (int *)malloc(sizeof(int) * (MATRIX_SIZE * MATRIX_SIZE / world_size));
    int *result = (int *)malloc(sizeof(int) * (MATRIX_SIZE * MATRIX_SIZE / world_size));

    if (world_rank == 0) {
        matrix_a = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
        matrix_b = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

        // Inicialização das matrizes
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                matrix_a[i * MATRIX_SIZE + j] = i + j;
                matrix_b[i * MATRIX_SIZE + j] = i - j;
            }
        }
        printf("Matriz A:\n");
        print_matrix(matrix_a);
        printf("Matriz B:\n");
        print_matrix(matrix_b);
    }

    // Distribuição das linhas da matriz A entre os processos
    MPI_Scatter(matrix_a, MATRIX_SIZE * MATRIX_SIZE / world_size, MPI_INT, submatrix_a,
                MATRIX_SIZE * MATRIX_SIZE / world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Transmissão da matriz B para todos os processos
    MPI_Bcast(matrix_b, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiplicação parcial das matrizes
    matrix_multiply(submatrix_a, matrix_b, result, MATRIX_SIZE / world_size);

    // Coleta das matrizes parciais e junção no processo 0
    MPI_Gather(result, MATRIX_SIZE * MATRIX_SIZE / world_size, MPI_INT, result,
               MATRIX_SIZE * MATRIX_SIZE / world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Impressão do resultado no processo 0
    if (world_rank == 0) {
        printf("Resultado:\n");
        print_matrix(result);
    }

    MPI_Finalize();
    free(submatrix_a);
    free(result);
    if (world_rank == 0) {
        free(matrix_a);
        free(matrix_b);
    }

    return 0;
}
