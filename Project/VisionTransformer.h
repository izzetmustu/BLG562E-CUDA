#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda.h>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::unique_lock, std::defer_lock
#include <chrono>
#include <algorithm>

// #define CHECK_CUDA(func) {\                                                                             
//     cudaError_t status = (func); \                                              
//     if (status != cudaSuccess) { \                                              
//         printf("CUDA API failed at line %d with error: %s (%d)\n", \            
//                __LINE__, cudaGetErrorString(status), status); \                 
//         return EXIT_FAILURE; \                                                  
//     } \                                                                          
// }

class ViT
{
private:
    std::mutex mtx;
    int numThread;
    std::vector<std::thread> threads;
    std::vector<std::thread> softmaxThreads;

    // Patch embeddings -> X->z0;  X=NxP^2.C, z0=(N+1)xD
    int numLayers;      // L
    int numPatches;     // N
    int hiddenSize;     // D
    int embeddingSize;  // P^2.C

    Eigen::MatrixXd xclass; // 1xD
    Eigen::MatrixXd E;      // (P^2.C)xD
    Eigen::MatrixXd Epos;   // (N+1)xD  

    // Multihead attention
    int numHeads;   
    std::vector<std::vector<Eigen::MatrixXd>> Wk;
    std::vector<std::vector<Eigen::MatrixXd>> Wq;
    std::vector<std::vector<Eigen::MatrixXd>> Wv;
    std::vector<Eigen::MatrixXd> Wy;
    Eigen::MatrixXd Y;

    // MLP
    int MLPSize;
    std::vector<std::vector<Eigen::MatrixXd>> MLPW;

    // CUDA stream
    cudaStream_t* streams;

public:
    ViT(int numLayers, int numPatches, int hiddenSize, int embeddingSize, int numHeads, int numThread, int MLPSize);
    Eigen::MatrixXd layerNorm(Eigen::MatrixXd* X);
    void softmaxDistributed(Eigen::MatrixXd* X, int rowID);
    void softmax(Eigen::MatrixXd* X);
    void softmaxGPU(Eigen::MatrixXd* X);
    void attentionCore(Eigen::MatrixXd* X, int levelID, int headID);
    void attentionCoreGPU(Eigen::MatrixXd* X, int levelID, int headID);
    Eigen::MatrixXd forward(Eigen::MatrixXd& X);
    Eigen::MatrixXd forwardGPU(Eigen::MatrixXd& X);
    void matMulWrapper(Eigen::MatrixXd* m1, Eigen::MatrixXd* m2, Eigen::MatrixXd* m3, int streamID);
};
