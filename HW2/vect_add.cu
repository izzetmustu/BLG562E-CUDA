#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048*2048
#define THREADS_PER_BLOCK 512

void checkCUDAError(const char*);
void random_ints(int *a);

// CPU
void vectorAddCPU(int *a, int *b, int *c){
// implement this part as the first step of the homework.
	int i;
	for(i = 0; i < N; i++)
	{
		c[i] = a[i] + b[i];
	}
}


int validate(int *a, int *ref){
	// implement this part to validate the GPU results, comparing the GPU generated results against the CPU
	//    generated reference values.
	// Required in steps 2a/2b/2c/2d

	int errors = 0;

	// if there is any mismatch between the vector elements, then increment the errors
	int i;
	for(i = 0; i < N; i++)
	{
		if(a[i] != ref[i])
		{
			errors++;
		}
	}

	return errors;
}

// 2
__global__ void vectorAdd(int* d_a, int* d_b, int* d_c, int n) {
	// implement this kernel accordingly for steps 2a/2b/2c/2d
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	// We do not want to overflow the array size
	if (index < n)
	{
		d_c[index] = d_a[index] + d_b[index];
	}

	// __syncthreads();  // Sync threads in a block
}

int main(void) {
    clock_t begin, end;
    double seconds;

	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size);
    random_ints(a);

	b = (int *)malloc(size);
    random_ints(b);

	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
    begin = clock();

    // Add cuda kernel launch code here for steps 2a/2b/2c/2d here.
	// 2.a
	// vectorAdd<<<1, 1024>>>(d_a, d_b, d_c, N);
	// 2.b
	// vectorAdd<<<N, 1>>>(d_a, d_b, d_c, N);
	// 2.c
	vectorAdd<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);	
	// 2.d
	// vectorAdd<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);	
    checkCUDAError("CUDA kernel");

    // Add code for device synchronization to make sure kernel has finished on GPU
	cudaDeviceSynchronize();	// Blocks CPU until CUDA calls have completed

    end = clock();
    seconds = (end - begin) / (double)CLOCKS_PER_SEC;

    printf("Vector addition kernel on GPU complete in %.2f seconds\n", seconds);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

    begin = clock();
    //perform CPU version
    vectorAddCPU(a, b, c_ref);
    end = clock();
    seconds = (end - begin) / (double)CLOCKS_PER_SEC;


    printf("Vector addition on CPU complete in %.2f seconds\n", seconds);

    //validate
    errors = validate(c, c_ref);
    printf("CUDA GPU result has %d errors.\n", errors);

	// Cleanup
	free(a);
    free(b);
    free(c);

	cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	checkCUDAError("CUDA cleanup");

	return 0;
}

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
		a[i] = rand();
	}
}
