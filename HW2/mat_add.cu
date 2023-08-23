#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define BLOCKS_PER_GRID_X N/THREADS_PER_BLOCK_X
#define BLOCKS_PER_GRID_Y N/THREADS_PER_BLOCK_Y

void checkCUDAError(const char*);
void random_ints(int *a);

// CPU
void matrixAddCPU(int *a, int *b, int *c){
	int i;
	int j;
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			c[i*N+j] = a[i*N+j] + b[i*N+j];
		}
	}
}

int validate(int *a, int *ref){
	int errors = 0;

	// if there is any mismatch between the matrix elements, then increment the errors
	int i;
	int j;
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			if(a[i*N+j] != ref[i*N+j])
			{
				errors++;
			}
		}
	}

	return errors;
}

// 3
__global__ void matrixAdd(int* d_a, int* d_b, int* d_c, int n) {
	// Find x and y indices 
	int index_x = threadIdx.x + blockDim.x*blockIdx.x;
	int index_y = threadIdx.y + blockDim.y*blockIdx.y;
	// We do not want to overflow the array size
	if  ((index_x < N) && (index_y < N))
	{
		d_c[index_y*n + index_x] = d_a[index_y*n + index_x] + d_b[index_y*n + index_x];
	}

	// __syncthreads();  // Sync threads in a block
}

int main(void) {
    clock_t begin, end;
    double seconds;

	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);
    random_ints(a);
    random_ints(b);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
    begin = clock();

    // Add cuda kernel launch code here
    dim3 grid(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, 1);
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);

	matrixAdd<<<grid, block>>>(d_a, d_b, d_c, N);	
    checkCUDAError("CUDA kernel");

    // Add code for device synchronization to make sure kernel has finished on GPU
	cudaDeviceSynchronize();	// Blocks CPU until CUDA calls have completed

    end = clock();
    seconds = (end - begin) / (double)CLOCKS_PER_SEC;

    printf("Matrix addition kernel on GPU complete in %.2f seconds\n", seconds);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

    begin = clock();
    //perform CPU version
    matrixAddCPU(a, b, c_ref);
    end = clock();
    seconds = (end - begin) / (double)CLOCKS_PER_SEC;


    printf("Matrix addition on CPU complete in %.2f seconds\n", seconds);

    //validate
    errors = validate(c, c_ref);
    printf("CUDA GPU result has %d errors.\n", errors);

	// Cleanup
	free(a);
    free(b);
    free(c);
	free(c_ref);

	cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	checkCUDAError("CUDA cleanup");

	return 0;
}





























// int main(void) {
//     clock_t begin, end;
//     double seconds;

// 	int **a, **b, **c, **c_ref;			// host copies of a, b, c
// 	int **d_a, **d_b, **d_c;			// device copies of a, b, c
// 	int errors;
// 	// unsigned int size = N * N * sizeof(int);


// 	// Alloc space for host copies of a, b, c and setup input values
// 	a = (int **)malloc(N * sizeof(int*));
// 	b = (int **)malloc(N * sizeof(int*));
// 	c = (int **)malloc(N * sizeof(int*));
// 	c_ref = (int **)malloc(N * sizeof(int*));

// 	for(int i = 0; i < N; i++)
// 	{
// 		a[i] = (int*)malloc(N * sizeof(int));
// 		b[i] = (int*)malloc(N * sizeof(int));
// 		c[i] = (int*)malloc(N * sizeof(int));
// 		c_ref[i] = (int*)malloc(N * sizeof(int));

//     	random_ints(a[i]);
//     	random_ints(b[i]);
// 	}

// 	// Alloc space for device copies of a, b, c
// 	int **d_a_tmp = (int **)malloc(N * sizeof(int*));
// 	int **d_b_tmp = (int **)malloc(N * sizeof(int*));
// 	int **d_c_tmp = (int **)malloc(N * sizeof(int*));
// 	for(int i = 0; i < N; i++)
// 	{
// 		// int *row;
// 		// cudaMalloc((void**)&row, N*sizeof(int));
// 		// cudaMemcpy(d_a + i, &row, sizeof(int*), cudaMemcpyHostToDevice);

// 		// int *row2;
// 		// cudaMalloc((void**)&row2, N*sizeof(int));
// 		// cudaMemcpy(d_a + i, &row2, sizeof(int*), cudaMemcpyHostToDevice);

// 		// int *row3;
// 		// cudaMalloc((void**)&row3, N*sizeof(int));
// 		// cudaMemcpy(d_a + i, &row3, sizeof(int*), cudaMemcpyHostToDevice);

// 		cudaMalloc((void**)&d_a_tmp[i], N*sizeof(int));
// 		cudaMalloc((void**)&d_b_tmp[i], N*sizeof(int));
// 		cudaMalloc((void**)&d_c_tmp[i], N*sizeof(int));
// 	}
// 	cudaMalloc((void***)&d_a, N*sizeof(int*));
// 	cudaMalloc((void***)&d_b, N*sizeof(int*));
// 	cudaMalloc((void***)&d_c, N*sizeof(int*));
// 	cudaMemcpy(d_a, d_a_tmp, N*sizeof(int), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_b, d_b_tmp, N*sizeof(int), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_c, d_c_tmp, N*sizeof(int), cudaMemcpyHostToDevice);


// 	checkCUDAError("CUDA malloc");

// 	// Copy inputs to device
// 	for(int i = 0; i < N; i++)
// 	{

// 		// int *row;
// 		// cudaMemcpy(row, d_a+i, sizeof(int*), cudaMemcpyDeviceToHost);
// 		// cudaMemcpy(a[i], row, sizeof(int) * N, cudaMemcpyDeviceToHost);

// 		// int *row2;
// 		// cudaMemcpy(&row2, d_b+i, sizeof(int*), cudaMemcpyDeviceToHost);
// 		// cudaMemcpy(b[i], row2, sizeof(int) * N, cudaMemcpyDeviceToHost);

// 		cudaMemcpy(*(d_a_tmp+i), *(a+i), N*sizeof(int), cudaMemcpyHostToDevice);
// 		cudaMemcpy(*(d_b_tmp+i), *(b+i), N*sizeof(int), cudaMemcpyHostToDevice);
// 	}
// 	checkCUDAError("CUDA memcpy");

// 	// Launch add() kernel on GPU
//     begin = clock();

//     // Add cuda kernel launch code here
//     dim3 grid(2048, 2048);
//     dim3 block(1, 1);

// 	matrixAdd<<<grid, block>>>(d_a, d_b, d_c, N);	
//     checkCUDAError("CUDA kernel");

//     // Add code for device synchronization to make sure kernel has finished on GPU
// 	cudaDeviceSynchronize();	// Blocks CPU until CUDA calls have completed

//     end = clock();
//     seconds = (end - begin) / (double)CLOCKS_PER_SEC;

//     printf("Matrix addition kernel on GPU complete in %.2f seconds\n", seconds);

// 	// Copy result back to host
// 	cudaMemcpy(c[0], d_c_tmp[0], N * sizeof(int), cudaMemcpyDeviceToHost);
// 	checkCUDAError("CUDA memcpy");

//     begin = clock();
//     //perform CPU version
//     matrixAddCPU(a, b, c_ref);
//     end = clock();
//     seconds = (end - begin) / (double)CLOCKS_PER_SEC;


//     printf("Matrix addition on CPU complete in %.2f seconds\n", seconds);

//     //validate
//     errors = validate(c, c_ref);
//     printf("CUDA GPU result has %d errors.\n", errors);

// 	// // Cleanup
// 	// for(int i = 0; i < N; i++)
// 	// {
// 	// 	free(a[i]);
// 	// 	free(b[i]);
// 	// 	free(c[i]);
// 	// 	free(c_ref[i]);

// 	// 	cudaFree(d_a[i]);
// 	// 	cudaFree(d_b[i]);
// 	// 	cudaFree(d_c[i]);
// 	// }

// 	// free(a);
//     // free(b);
//     // free(c);
// 	// free(c_ref);

// 	// cudaFree(d_a);
//     // cudaFree(d_b);
//     // cudaFree(d_c);

// 	checkCUDAError("CUDA cleanup");

// 	return 0;
// }










void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < N; j++){
			a[i*N + j] = rand();
		}
	}
}


