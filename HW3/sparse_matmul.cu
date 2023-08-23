#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
// #include <cuda/semaphore>

typedef float elemType;

template<class T>
struct Entry {
    int row, col;
    T value;
    Entry() 
    {
        row = 0;
        col = 0;
    }
    Entry(int r, int c, T v) : row(r), col(c), value(v) {}
};

template<class T>
bool compareCOO(const Entry<T>& A, const Entry<T>& B)
{
    if(A.row != B.row)
    {
        return A.row < B.row;
    }
    else
    {
        return A.col < B.col;
    }
}

template<class T>
class SparseMatrixCOO
{
public:
    std::vector<Entry<T>> matrix;
public:
    SparseMatrixCOO(){}
    SparseMatrixCOO<T> getTranspose() const
    {
        SparseMatrixCOO<T> transposed;
        for(typename std::vector<Entry<T>>::const_iterator  it = matrix.begin(); it != matrix.end(); it++)
        {
            transposed.matrix.push_back(Entry<T>(it->col, it->row, it->value));
        } 
        std::sort(transposed.matrix.begin(), transposed.matrix.end(), compareCOO<elemType>);

        return transposed;
    }
    bool isNonZero(int r, int c) const
    {
        for(typename std::vector<Entry<T>>::const_iterator  it = matrix.begin(); it != matrix.end(); it++)
        {
            if(it->row == r && it->col == c)
            {
                return true;
            }
            if(it->row > r)
            {
                break;
            }
        }
        return false;
    }
    void increase(int r, int c, int v)
    {
        for(typename std::vector<Entry<T>>::iterator  it = matrix.begin(); it != matrix.end(); it++)
        {
            if(it->row == r && it->col == c)
            {
                it->value += v; 
            }
        }
    }
    ~SparseMatrixCOO(){}
};

template<class T>
class SparseMatrixCSR
{
public:
    std::vector<int>vi, vj;
    std::vector<T> values;
public:
    SparseMatrixCSR(){}
    ~SparseMatrixCSR(){}
};

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

template<class T>
__global__ void sparseMatMul(Entry<T>* m1, Entry<T>* m2, T* m33, int s1, int s2, int N, int M)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx < s1)
    {
        for(int i = 0; i < s2; i++)
        {
            if(m1[idx].col == m2[i].row)
            {
                m33[idx*N*M + m1[idx].row*N + m2[i].col] += m1[idx].value*m2[i].value;
            }
        }
    }
}

template<class T>
__global__ void mergeResults(T* seperated, T* merged, int NM, int S1)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < NM)
    {
        T result = 0;
        for(int i = 0; i < S1; i++)
        {
            result += seperated[NM*i + idx];
        }
        merged[idx] = result;
    }

}

template<class T>
void CPUSparseMatMul(const SparseMatrixCOO<T>& m1, const SparseMatrixCOO<T>& m2, SparseMatrixCOO<T>* m3)
{
    SparseMatrixCOO<T> transposed = m2.getTranspose();

    for(typename std::vector<Entry<T>>::const_iterator  it = m1.matrix.begin(); it != m1.matrix.end(); it++)
    {
        for(typename std::vector<Entry<T>>::const_iterator  jt = transposed.matrix.begin(); jt != transposed.matrix.end(); jt++)
        {
            if(it->col == jt-> col)
            {
                if(!m3->isNonZero(it->row, jt->row))
                {
                    m3->matrix.push_back(Entry<T>(it->row, jt->row, it->value * jt->value));
                }
                else
                {
                    m3->increase(it->row, jt->row, it->value * jt->value);
                }
            }
        }
    }
}

template<class T>
void dense2Sparse(elemType* src, SparseMatrixCOO<T>& dst, int N, int M)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            T x = src[i*N + j];
            if(x != 0)
            {
                dst.matrix.push_back(Entry<T>(i, j, x));
            }
        }
    }
}

template<class T>
bool sparseCompare(const SparseMatrixCOO<T>& m1, const SparseMatrixCOO<T>& m2)
{
    std::cout << "Comparing, m1 size: " << m1.matrix.size() << " m2 size: " << m2.matrix.size() << std::endl;
    if(m1.matrix.size() != m2.matrix.size())
    {
        return false;
    }

    for(int i = 0; i < m1.matrix.size(); i++)
    {
        if((m1.matrix[i].row != m2.matrix[i].row) || (m1.matrix[i].col != m2.matrix[i].col) || (m1.matrix[i].value/m2.matrix[i].value < 0.8 || m1.matrix[i].value/m2.matrix[i].value > 1.2))
        {
            std::cout << "i: " << i << std::endl;
            std::cout << m1.matrix[i].row << " " <<  m1.matrix[i].col << " " << m1.matrix[i].value << " || " << m2.matrix[i].row << " " << m2.matrix[i].col << " " << m2.matrix[i].value << std::endl;
            return false;
        }
    }
    return true;
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

template<class T>
void printSparse(const SparseMatrixCOO<T>& m)
{
    std::cout << m.matrix.size() << std::endl;
    for(typename std::vector<Entry<T>>::const_iterator it = m.matrix.begin(); it != m.matrix.end(); it++)
    {
        std::cout << it->row << " " << it->col << " " << it->value << std::endl;
    }
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
int readFileCOO(std::string fileName, SparseMatrixCOO<T> &src)
{
    std::string delimiter = "_";
    std::string token = fileName.substr(0, fileName.find(delimiter)); // token is matrix width
    int N = std::stoi(token);

    std::ifstream fin(fileName);
    int i, j;
    T v;

    while(fin.peek() == '%') fin.ignore(2048, '\n');

    while (fin >> i >> j >> v)
    {
        src.matrix.push_back(Entry<T>(i - 1, j - 1, v));
    }

    fin.close();

    return N;
}

int main(int argc, char **argv)
{
    clock_t  begin, end;

    SparseMatrixCOO<elemType> coo1;
    SparseMatrixCOO<elemType> coo2;
    SparseMatrixCOO<elemType> coo3CPU;

    int N = readFileCOO<elemType>(argv[1], coo1);
    int M = readFileCOO<elemType>(argv[2], coo2);
    int S1 = coo1.matrix.size();
    int S2 = coo2.matrix.size();
    std::cout << "S1: " << S1 << " S2: " << S2 << std::endl;
    // printSparse(coo1);

    // SORT ENTRIES
    std::sort(coo1.matrix.begin(), coo1.matrix.end(), compareCOO<elemType>);
    std::sort(coo2.matrix.begin(), coo2.matrix.end(), compareCOO<elemType>);
    
    begin = clock();
    CPUSparseMatMul<elemType>(coo1, coo2, &coo3CPU);
    end = clock();
    printf("Sparse(square with size %d X %d) matmul on CPU complete in %.5f seconds\n", N, M, (end - begin) / (double)CLOCKS_PER_SEC);

    Entry<elemType> *m1, *m2;
    elemType *m3, *m33;
    size_t size1 = coo1.matrix.size()*sizeof(Entry<elemType>);
    size_t size2 = coo2.matrix.size()*sizeof(Entry<elemType>);
    size_t size3 = N*M*sizeof(elemType);    
    size_t size33 = S1*N*M*sizeof(elemType);    
    
    cudaMallocManaged(&m1, size1);
    cudaMallocManaged(&m2, size2);
    cudaMallocManaged(&m3, size3);
    cudaMallocManaged(&m33, size33);
    checkCUDAError("CUDA malloc managed");
    std::copy(coo1.matrix.begin(), coo1.matrix.end(), m1);
    std::copy(coo2.matrix.begin(), coo2.matrix.end(), m2);

    // Assign kernel size
    int ttt = 32;
    dim3 grid((N + ttt - 1) / ttt, (M + ttt - 1) / ttt, 1);
    dim3 block(ttt, ttt, 1);
    
    std::cout << "Init kernel grid: " << grid.x << " block: " << block.x << std::endl; 
    begin = clock();
    initMatrix<<<grid, block>>>(m3, N, M, 1, 0);
    checkCUDAError("CUDA init kernel");
    cudaDeviceSynchronize();
    end = clock();
    printf("CUDA init kernel on GPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC);    

    // Assign kernel size
    ttt = 32;
    grid = dim3((S1 + ttt - 1) / ttt, 1, 1);
    block = dim3(ttt, 1, 1);

    std::cout << "Sparse matmul kernel grid: " << grid.x << " block: " << block.x << std::endl; 
    begin = clock();
    sparseMatMul<<<grid, block>>>(m1, m2, m33, S1, S2, N, M);
    checkCUDAError("CUDA sparse matmul kernel");
    cudaDeviceSynchronize();
    end = clock();
    printf("Sparse matmul kernel on GPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC);       
    // printMatrix(m33, S1, N*M);

    // Assign kernel size
    ttt = 32;
    grid = dim3((N*M + ttt - 1) / ttt, 1, 1);
    block = dim3(ttt, 1, 1);
    std::cout << "Merge kernel grid: " << grid.x << " block: " << block.x << std::endl; 
    begin = clock();
    mergeResults<<<grid, block>>>(m33, m3, N*M, S1);
    checkCUDAError("CUDA merge matmul kernel");
    cudaDeviceSynchronize();
    end = clock();
    printf("Sparse merge matmul kernel on GPU complete in %.5f seconds\n", (end - begin) / (double)CLOCKS_PER_SEC); 
    // printMatrix(m3, N, M);

    SparseMatrixCOO<elemType> m3Sparse;
    begin = clock();
    dense2Sparse<elemType>(m3, m3Sparse, N, M);
    end = clock();
    printf("Dense(square with size %d X %d) to sparse on CPU complete in %.5f seconds\n", N, M, (end - begin) / (double)CLOCKS_PER_SEC);  
  
    // Sort to compare
    std::sort(coo3CPU.matrix.begin(), coo3CPU.matrix.end(), compareCOO<elemType>);
    std::cout << sparseCompare(coo3CPU, m3Sparse) << std::endl;

    return 0;
}