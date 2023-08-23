#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>

#define N 2048
#define M 2048

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define BLOCKS_PER_GRID_X N/THREADS_PER_BLOCK_X
#define BLOCKS_PER_GRID_Y N/THREADS_PER_BLOCK_Y

typedef float elemType;

// nxm matrix src is filled with the supplied value
__global__ void initMatrix(elemType* src, size_t n, size_t m, size_t width, elemType value = 0)
{
	// Find x and y indices 
	int column = threadIdx.x + blockDim.x*blockIdx.x;
	int row = threadIdx.y + blockDim.y*blockIdx.y;

    if((column < m/width) && (row < n/width))
    {
        size_t i;
        size_t j;
        for (i = 0; i < width; i++)
        {
            for (j = 0; j < width; j++)
            {
                src[(row*width + i)*m + column*width + j] = value;
            }
        }
    }
}

__global__ void matrixMul(elemType* m1, elemType* m2, elemType* m3, int Width) 
{
    // Calculate the column index of m3 and m2
    int column = blockIdx.x*blockDim.x+threadIdx.x;
    // Calculate the row index of the m3 element and m1
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    if ((row < Width) && (column < Width)) {
        elemType result = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k)
        {
            result += m1[row*Width+k]*m2[k*Width+column];
        }
        m3[row*Width+column] = result;
    }
}

void CPUMatrixMul(elemType* m1, elemType* m2, elemType* m3)
{
    size_t i, j, k;
    elemType local;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            local = 0;
            for (k = 0; k < N; k++)
            {
                local += m1[i*N + k] * m2[k*N + j];
            }
            m3[i*N + j] = local;
        }
    }
}

int validate(elemType* m1, elemType* m2){
	int errors = 0;

	// if there is any mismatch between the matrix elements, then increment the errors
	int i;
	int j;
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			if(m1[i*N+j] != m2[i*N+j])
			{
				errors++;
			}
		}
	}

	return errors;
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

template<class T>
void random_ints(T *a, size_t n, size_t m)
{
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			a[i*N + j] = rand() % 100;
		}
	}
}


void printMatrix(elemType* matrix, size_t n, size_t m)
{
    size_t i, j;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < m; j++)
        {
            std::cout << matrix[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    clock_t  begin, end;

    elemType *m1, *m2, *m3, *correct;
    size_t size1 = N*N*sizeof(elemType);
    size_t size2 = M*M*sizeof(elemType);
    size_t size3 = N*M*sizeof(elemType);

    cudaMallocManaged(&m1, size1);
    cudaMallocManaged(&m2, size2);
    cudaMallocManaged(&m3, size3);
    cudaMallocManaged(&correct, size3);

    // INIT MATRICES
    begin = clock();
    random_ints<elemType>(m1, N, N);
    random_ints<elemType>(m2, M, M);
    end = clock();
    printf("Matrix init on CPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC);

    // Assign kernel size
    dim3 grid(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, 1);
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    
    begin = clock();
    initMatrix<<<grid, block>>>(m3, N, M, 1, 0);
    checkCUDAError("CUDA init kernel");
    cudaDeviceSynchronize();
    end = clock();
    printf("CUDA init kernel on GPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC);

    // MATMUL
    begin = clock();
    matrixMul<<<grid, block>>>(m1, m2, m3, N);
    checkCUDAError("CUDA dense matmul kernel");
    cudaDeviceSynchronize();
    end = clock();
    printf("CUDA dense matmul kernel on GPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC);

    // VALIDATE
    begin = clock();
    CPUMatrixMul(m1, m2, correct);
    end = clock();
    printf("Dense matmul kernel on CPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC);

    int errors = validate(m3, correct);
    printf("CUDA GPU result has %d errors.\n", errors);
    // printMatrix(correct, N, N);

    // Free memory
    cudaFree(m1);
    cudaFree(m2);
    cudaFree(m3);
    cudaFree(correct);

	checkCUDAError("CUDA cleanup");

    return 0;
}



// // NbyN X MbyM = NbyM
// __global__ void tiledDenseMatMul(elemType* m1, elemType* m2, elemType* m3, int Width)
// {
//     __shared__ elemType partm1[TILE_WIDTH][TILE_WIDTH];
//     __shared__ elemType partm2[TILE_WIDTH][TILE_WIDTH];

//     int bx = blockIdx.x; int by = blockIdx.y;
//     int tx = threadIdx.x; int ty = threadIdx.y;

// 	// Find x and y indices 
// 	int row = ty + blockDim.y*by;
// 	int column = tx + blockDim.x*bx;

//     elemType result = 0;

//     // Loop over the M and N tiles required to compute the P element
//     for (int i = 0; i < (Width-1)/TILE_WIDTH + 1; ++i) 
//     {
//         // Collaborative loading of M and N tiles into shared memory
//         partm1[ty][tx] = m1[row*Width + i*TILE_WIDTH+tx];
//         partm2[ty][tx] = m2[(i*TILE_WIDTH+ty)*Width + column];
//         __syncthreads();

//         for (int j = 0; j < TILE_WIDTH; ++j)
//         {
//             result += partm1[ty][j]*partm2[j][tx];
//         }
//         __syncthreads();
//     }
//     m3[row*Width+column] = result;
    
// }