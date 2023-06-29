#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "chrono.c"

#define MAX_DOUBLE 100000000.00

int nla, ncb, m;

double *geraMatriz(int nl, int nc){
   double *resultado = malloc(nl * nc * sizeof(double));
   for(int x = 0; x < nl*nc; x++)
      resultado[x] = (double)random() / MAX_DOUBLE;
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

int main(int argc, char *argv[]){

   if (argc < 4){
      printf("usage: mpriun -np <np> %s <Nla> <M> <Ncb> (-v)\n", argv[0]);
      return 0;
   }
   nla = atoi(argv[1]);
   m   = atoi(argv[2]);
   ncb = atoi(argv[3]);

   double *a = geraMatriz(nla, m);
   double *b = geraMatriz(m, ncb);
   double *c = malloc(nla * ncb * sizeof(double));

   multMatriz(a, b, c, nla, m, ncb);

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
