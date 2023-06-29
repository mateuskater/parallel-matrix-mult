#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "chrono.c"

#define DOUBLE_DIVIDE 100000000.00

int nla, ncb, m;
int nproc, rank;

double *geraMatriz(int nl, int nc){
   double *resultado = malloc(nl * nc * sizeof(double));
   for(int x = 0; x < nl*nc; x++)
      resultado[x] = (double)random() / DOUBLE_DIVIDE;
   return resultado;
}

// Salva as cordenadas nos ponteiros passados
void indiceToCord(int indice, int nl, int nc, int *i, int *j){
   *i = indice / nc;
   *j = indice % nc;
}

// Transforma o a cordenada (i,j) em um indice
int cordToIndice(int i, int j, int nl, int nc){
   return i * nc + j;
}

// Multiplica matriz A pela B e salva na C
void multMatriz(double *A, double *B, double *C, int nla, int m, int ncb){
   int i, j;
   for (int x = 0; x < nla * ncb; x++){
      indiceToCord(x, nla, ncb, &i, &j);
      C[x] = 0;
      for (int y = 0; y < m; y++){
         C[x] += A[cordToIndice(i, y, nla, m)] * B[cordToIndice(y, j, m, ncb)];
      }
   }
}

void parallelMultMatriz(double *A, double *B, double *C, int nla, int m, int ncb){
   // Alocar as matrizes/vetores temporários de cada thread
   double *a = malloc(m * sizeof(double)); // uma linha de A
   double *b = B;
   if (rank != 0)
      b = malloc(m * ncb * sizeof(double)); // a matriz B
   double *c = malloc(ncb * sizeof(double)); // uma linha de C

   if (a==NULL || b==NULL || c==NULL){
      fprintf(stderr, "Could not alocate matrixes, my rank: %d\n");
      return;
   }

   // Broadcast da matriz b
   MPI_Bcast(b, m * ncb, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   for (int fase = 0; fase < nla / nproc; fase++){
      int deslocamentoA = m * nproc * fase;
      int deslocamentoC = ncb * nproc * fase;
      MPI_Scatter(A + deslocamentoA, m, MPI_DOUBLE,
            a, m, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

      multMatriz (a, b, c, 1, m, ncb);

      MPI_Gather (c, ncb, MPI_DOUBLE,
            C + deslocamentoC, ncb, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
   }
}

int main(int argc, char *argv[]){

   if (argc < 4){
      printf("usage: mpriun -np <np> %s <Nla> <M> <Ncb> (-v)\n", argv[0]);
      return 0;
   }
   nla = atoi(argv[1]);
   m   = atoi(argv[2]);
   ncb = atoi(argv[3]);

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nproc);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   double *a = NULL;
   double *b = NULL;
   double *c = NULL;

   if (rank == 0){
      a = geraMatriz(nla, m);
      b = geraMatriz(m, ncb);
      c = malloc(nla * ncb * sizeof(double));
      if (a==NULL || b==NULL || c==NULL){
         fprintf(stderr, "Could not alocate matrixes\n");
         return 0;
      }
   }

   parallelMultMatriz(a, b, c, nla, m, ncb);

   if (rank == 0){
      printf("a = [\n");
      for (int i = 0; i < nla; i++) {
         for (int j = 0; j < m; j++) 
            printf("\t%0.2f", a[cordToIndice(i, j, nla, m)]);
         printf("\n");
      }
      printf("]\n");

      printf("b = [\n");
      for (int i = 0; i < m; i++) {
         for (int j = 0; j < ncb; j++) 
            printf("\t%0.2f", b[cordToIndice(i, j, m, ncb)]);
         printf("\n");
      }
      printf("]\n");

      printf("c = [\n");
      for (int i = 0; i < nla; i++) {
         for (int j = 0; j < ncb; j++) 
            printf("\t%0.2f", c[cordToIndice(i, j, nla, ncb)]);
         printf("\n");
      }
      printf("]\n");
   }

   MPI_Finalize();

}
