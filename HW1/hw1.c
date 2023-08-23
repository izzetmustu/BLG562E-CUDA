
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 1024

// Default
typedef double elem_t;
// typedef float elem_t;

typedef elem_t **matrixNN;

void init_random(matrixNN m){
  // initialize a given matrix with random numbers
  int i, j;
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      m[i][j] = rand() / (elem_t)RAND_MAX;
    }
  }
}

void init_empty(matrixNN m){
  // initialize a given matrix with value of 0
  int i, j;
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      m[i][j] = 0;
    }
  }
}

// // Default
// void mat_mul(matrixNN r, const matrixNN a, const matrixNN b) {
//   // dot product matrix multiplication
//   int i, j, k;
  
//   for (i = 0; i < N; i++){
//     for (j = 0; j < N; j++){
//       r[i][j] = 0;
//       for (k = 0; k < N; k++){
// 	r[i][j] += a[i][k] * b[k][j];
//       }
//     }
//   }
// }

// // 2.c
// void mat_mul(matrixNN r, const matrixNN a, const matrixNN b) {
//   // dot product matrix multiplication
//   int i, j, k;
  
//   for (i = 0; i < N; i++){
//     for (j = 0; j < N; j++){
//       elem_t local = 0; // local variable
//       for (k = 0; k < N; k++){
// 	      local += a[i][k] * b[k][j];
//       }
//       r[i][j] = local;
//     }
//   }
// }

void transpose(matrixNN A){
  int i, j;
  
  for(i = 0; i < N; i++){
    for(j = i+1; j<N; j++){
        int temp = A[j][i];
        A[j][i] = A[i][j];
        A[i][j] = temp;
    }
  }
}

// // 2.d
// void mat_mul(matrixNN r, const matrixNN a, const matrixNN b) {
//   // dot product matrix multiplication
//   int i, j, k;
//   // Transpose second matrix
//   transpose(b);

//   for (i = 0; i < N; i++){
//     for (j = 0; j < N; j++){
//       elem_t local = 0; // local variable
//       for (k = 0; k < N; k++){
// 	      local += a[i][k] * b[j][k];
//       }
//       r[i][j] = local;
//     }
//   }
// }

// // 2.e
// void mat_mul(matrixNN r, const matrixNN a, const matrixNN b) {
//   // dot product matrix multiplication
//   int i, j, k;
  
//   for (i = 0; i < N; i++){
//     for (j = 0; j < N; j++){
//       elem_t local = 0; // local variable
//       for (k = 0; k < N; k += 8){
// 	      local += a[i][k] * b[k][j];
// 	      local += a[i][k+1] * b[k+1][j];
// 	      local += a[i][k+2] * b[k+2][j];
// 	      local += a[i][k+3] * b[k+3][j];
// 	      local += a[i][k+4] * b[k+4][j];
// 	      local += a[i][k+5] * b[k+5][j];
// 	      local += a[i][k+6] * b[k+6][j];
// 	      local += a[i][k+7] * b[k+7][j];
//       }
//       r[i][j] = local;
//     }
//   }
// }

// // 2.f
// void mat_mul(matrixNN r, const matrixNN a, const matrixNN b) {
//   // dot product matrix multiplication
//   int i, j, k;
  
//   for (i = 0; i < N; i++){
//     for (j = 0; j < N; j++){
//       r[i][j] = 0;
//       for (k = 0; k < N; k++){
// 	      r[i][j] += a[i][k] * b[k][j];
//       }
//     }
//   }
// }

// 3
void mat_mul(matrixNN r, const matrixNN a, const matrixNN b) {
  // dot product matrix multiplication
  int i, j, k;
  
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      for (k = 0; k < N; k++){
	      r[j][k] += a[j][i] * b[i][k];
      }
    }
  }
}

void write_to_file(const char *filename, const matrixNN r){
  FILE *f;
  int i, j;
  
  f = fopen(filename, "w");
  if (f == NULL){
    fprintf(stderr, "Error opening file '%s' for write\n", filename);
    return;
  }
  
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      fprintf(f, "%0.2f\t", r[i][j]);
    }
    fprintf(f, "\n");
  }
}


void main(){
	clock_t begin, end;
	double seconds;

	int i;


	matrixNN a = (matrixNN)malloc(sizeof(elem_t*)*N);
	for (i = 0; i < N;i++)
		a[i] = (elem_t*)malloc(sizeof(elem_t)*N);
	matrixNN b = (matrixNN)malloc(sizeof(elem_t*)*N);
	for (i = 0; i < N; i++)
		b[i] = (elem_t*)malloc(sizeof(elem_t)*N);
	matrixNN c = (matrixNN)malloc(sizeof(elem_t*)*N);
	for (i = 0; i < N; i++)
		c[i] = (elem_t*)malloc(sizeof(elem_t)*N);

	init_random(a);
	init_random(b);
	init_empty(c);

	begin = clock(); // get the clock just before mat_mul function

	mat_mul(c, a, b);

	end = clock(); // read the clock at the end of mat_mul function
	seconds = (end - begin) / (double)CLOCKS_PER_SEC;


	printf("Matrix multiply with float type complete in %.2f seconds\n", seconds);
	
	write_to_file("matrix_mul.txt", c);
	printf("Done writing results\n");

	for (i = 0; i < N; i++)
		free(a[i]);
	free(a);
	for (i = 0; i < N; i++)
		free(b[i]);
	free(b);
	for (i = 0; i < N; i++)
		free(c[i]);
	free(c);
}

