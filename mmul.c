#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "chrono.c"

#define DOUBLE_DIVIDE 100000000.00

int nla, ncb, m;
int nproc, rank;
chronometer_t mmultChrono;
char hostName[MPI_MAX_PROCESSOR_NAME];

double *geraMatriz(int nl, int nc){
   double *resultado = (double*)malloc(nl * nc * sizeof(double));
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
   double *a = (double*)malloc(m * sizeof(double)); // uma linha de A
   double *b = B;
   if (rank != 0)
      b = (double*)malloc(m * ncb * sizeof(double)); // a matriz B
   double *c = (double*)malloc(ncb * sizeof(double)); // uma linha de C

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
      c = (double*)malloc(nla * ncb * sizeof(double));
      if (a==NULL || b==NULL || c==NULL){
         fprintf(stderr, "Could not alocate matrixes\n");
         return 0;
      }
   }

   if (rank == 0) {
      chrono_reset(&mmultChrono);
      chrono_start(&mmultChrono);
   }

   parallelMultMatriz(a, b, c, nla, m, ncb);

   MPI_Barrier(MPI_COMM_WORLD);

   int resultLen;
   MPI_Get_processor_name(hostName, &resultLen);
   printf("Process %d ran on host %s\n", rank, hostName);

   if(rank == 0){
      chrono_stop(&mmultChrono);
      chrono_reportTime(&mmultChrono, "mmultChrono");

      // calcular e imprimir a VAZAO (nesse caso: numero de BYTES/s)
      double total_time_in_seconds = (double)chrono_gettotal(&mmultChrono) /
         ((double)1000 * 1000 * 1000);
      double total_time_in_micro = (double)chrono_gettotal(&mmultChrono) /
         ((double)1000);
      printf("total_time_in_seconds: %lf s\n", total_time_in_seconds);
      double GFLOPS = (((double)nla*m*ncb) / ((double)total_time_in_seconds*1000*1000*1000));
      printf("Throughput: %lf GFLOPS\n", GFLOPS*(nproc-1));
   }

   // se for -v
   if (argc > 4 && !strcmp(argv[4], "-v") && rank == 0) {
      double *d = (double*)malloc(nla * ncb * sizeof(double));
      multMatriz(a, b, d, nla, m, ncb);
      int i, j;
      for (int x = 0; x < nla * ncb; x++) {
         if (c[x] != d[x]) {
            indiceToCord(x, nla, ncb, &i, &j);
            fprintf(stderr, "ERROR ON ELEMENT (%d, %d) OF THE MATRIX\n", i, j);
            fprintf(stderr, "C[%d][%d] is %0.2f and should be %0.2f\n",
                  i, j, c[x], d[x]);
         }
      }
   }
   MPI_Finalize();

}
