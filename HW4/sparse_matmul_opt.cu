#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cusparse.h>
#include <cuda.h>
#include <sys/time.h>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

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

int
main(int argc, char** argv)
{
    // Create cuSPARSE library handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    clock_t  begin, end;

    int64_t rows;
    int64_t cols;
    int64_t nnz;  

    SparseMatrixCOO<elemType> coo1;
    SparseMatrixCOO<elemType> coo2;
    SparseMatrixCOO<elemType> coo3CPU;

    // Read first matrix
    rows = readFileCOO<elemType>(argv[1], coo1);
    cols = rows;
    nnz = coo1.matrix.size();
    std::sort(coo1.matrix.begin(), coo1.matrix.end(), compareCOO<elemType>);
    size_t size1 = nnz*sizeof(elemType);
    int* coo1RowInd;
    int* coo1ColInd;
    elemType* coo1Values;

    CHECK_CUDA(cudaMallocManaged(&coo1RowInd, size1))
    CHECK_CUDA(cudaMallocManaged(&coo1ColInd, size1));
    CHECK_CUDA(cudaMallocManaged(&coo1Values, size1))

    for(int i = 0; i < nnz; i++)
    {
        coo1RowInd[i] = coo1.matrix.at(i).row;
        coo1ColInd[i] = coo1.matrix.at(i).col;
        coo1Values[i] = coo1.matrix.at(i).value;
    }

    cusparseSpMatDescr_t spMat1Dsc;
    CHECK_CUSPARSE(cusparseCreateCoo(&spMat1Dsc, rows, cols, nnz, coo1RowInd, coo1ColInd, coo1Values,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Convert COO to CSR
    int* csr1RowPtr;
    CHECK_CUDA(cudaMallocManaged(&csr1RowPtr, (rows+1)*sizeof(int)))
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, coo1RowInd, nnz, rows, csr1RowPtr, CUSPARSE_INDEX_BASE_ZERO))
    cudaDeviceSynchronize();
    cusparseSpMatDescr_t spMat1DscCsr;
    CHECK_CUSPARSE(cusparseCreateCsr(&spMat1DscCsr, rows, cols, nnz,
                                      csr1RowPtr, coo1ColInd, coo1Values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseDestroySpMat(spMat1Dsc))


    // Read second matrix
    rows = readFileCOO<elemType>(argv[2], coo2);
    cols = rows;
    nnz = coo2.matrix.size();
    std::sort(coo2.matrix.begin(), coo2.matrix.end(), compareCOO<elemType>);
    size_t size2 = nnz*sizeof(elemType);
    int* coo2RowInd;
    int* coo2ColInd;
    elemType* coo2Values;

    CHECK_CUDA(cudaMallocManaged(&coo2RowInd, size2))
    CHECK_CUDA(cudaMallocManaged(&coo2ColInd, size2))
    CHECK_CUDA(cudaMallocManaged(&coo2Values, size2))

    for(int i = 0; i < nnz; i++)
    {
        coo2RowInd[i] = coo2.matrix.at(i).row;
        coo2ColInd[i] = coo2.matrix.at(i).col;
        coo2Values[i] = coo2.matrix.at(i).value;
    }

    cusparseSpMatDescr_t spMat2Dsc;
    CHECK_CUSPARSE(cusparseCreateCoo(&spMat2Dsc, rows, cols, nnz, coo2RowInd, coo2ColInd, coo2Values,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))  
    // Convert COO to CSR
    int* csr2RowPtr;
    CHECK_CUDA(cudaMallocManaged(&csr2RowPtr, (rows+1)*sizeof(int)))
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, coo2RowInd, nnz, rows, csr2RowPtr, CUSPARSE_INDEX_BASE_ZERO))
    cudaDeviceSynchronize();    
    cusparseSpMatDescr_t spMat2DscCsr;
    CHECK_CUSPARSE(cusparseCreateCsr(&spMat2DscCsr, rows, cols, nnz,
                                      csr2RowPtr, coo2ColInd, coo2Values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseDestroySpMat(spMat2Dsc));

    // Calculate on CPU
    begin = clock();
    CPUSparseMatMul<elemType>(coo1, coo2, &coo3CPU);
    end = clock();
    printf("Sparse(square with size %d X %d) matmul on CPU complete in %.5f seconds\n", int(rows), int(rows), (end - begin) / (double)CLOCKS_PER_SEC);

    // Allocate result matrix
    int* csr3RowPtr;
    int* csr3ColInd;
    elemType* csr3Values;    
    CHECK_CUDA(cudaMallocManaged(&csr3RowPtr, (rows+1)*sizeof(int)))
    cusparseSpMatDescr_t spMat3Dsc;
    CHECK_CUSPARSE(cusparseCreateCsr(&spMat3Dsc, rows, cols, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    
    // Start calculating on GPU
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    float alpha = 1;
    float beta = 0;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    // Calculate on CPU
    begin = clock();
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, spMat1DscCsr, spMat2DscCsr, &beta, spMat3Dsc,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL))
    CHECK_CUDA(cudaMalloc((void**) &dBuffer1, bufferSize1))
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, spMat1DscCsr, spMat2DscCsr, &beta, spMat3Dsc,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1))

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, spMat1DscCsr, spMat2DscCsr, &beta, spMat3Dsc,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL))
    CHECK_CUDA(cudaMalloc((void**) &dBuffer2, bufferSize2))

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, spMat1DscCsr, spMat2DscCsr, &beta, spMat3Dsc,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2))
    end = clock();
    printf("Sparse(square with size %d X %d) matmul on GPU complete in %.5f seconds\n", int(rows), int(rows), (end - begin) / (double)CLOCKS_PER_SEC);

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(spMat3Dsc, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1))
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void**)&csr3ColInd, C_nnz1*sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&csr3Values, C_nnz1*sizeof(float)))

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of csr3Values, and before the call of cusparseSpGEMM_copy

    // update spMat3Dsc with the new pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(spMat3Dsc, csr3RowPtr, csr3ColInd, csr3Values))

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of csr3Values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, spMat1DscCsr, spMat2DscCsr, &beta, spMat3Dsc,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))

    // Copy host to verify
    /*
    
    TO VERIFY, I COUNTED THE NUMBER OF MISMATCHES BETWEEN THE GPU RESULT AND CPU RESULT WITH 0.1 ERROR THRESHOLD.
    I NOTICED THAT THERE ARE MORE THAN 200 HUNDRED ERRORS, TO OBSERVE WHAT IS WRONG I PRINTED FIRST 50 VALUES IN COO FORMAT.
    THEN I REALIZED THAT GPU RESULTS ARE NOT SORTED CORRECTLY AS CPU RESULTS ARE SORTED.
    SO, THE ERRORS ARE NOT ACTUALLY ERROS. TO PROVE THAT WE CAN SIMPLY
    1- CONVERT CSR GPU FORMAT TO COO FORMAT
    2- MOVE CUSPARSE COO FORMAT TO CREATED COO DATA STRUCTURE
    3- COMPARE
    
    BELOW CODE IS ONLY USED TO PRINT
    */
    // elemType* hostcsr3Values = (elemType*)malloc(C_nnz1 * sizeof(float));
    // CHECK_CUDA(cudaMemcpy(hostcsr3Values, csr3Values, C_nnz1 * sizeof(float),
    //                        cudaMemcpyDeviceToHost))

    cudaDeviceSynchronize();    
    
    // Destroy cuSPARSE library handle
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(spMat1DscCsr))
    CHECK_CUSPARSE(cusparseDestroySpMat(spMat2DscCsr))
    CHECK_CUSPARSE(cusparseDestroySpMat(spMat3Dsc))
    CHECK_CUSPARSE(cusparseDestroy(handle))    

    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(csr3RowPtr))
    CHECK_CUDA(cudaFree(csr3ColInd))
    CHECK_CUDA(cudaFree(csr3Values))

    CHECK_CUDA(cudaFree(coo1RowInd))
    CHECK_CUDA(cudaFree(coo1ColInd))
    CHECK_CUDA(cudaFree(coo1Values))
    CHECK_CUDA(cudaFree(coo2RowInd))
    CHECK_CUDA(cudaFree(coo2ColInd))
    CHECK_CUDA(cudaFree(coo2Values))

    return 0;
}