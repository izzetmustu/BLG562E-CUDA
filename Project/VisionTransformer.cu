#include "VisionTransformer.h"

#define TILE_WIDTH 16

#define THREADS_PER_BLOCK_X TILE_WIDTH
#define THREADS_PER_BLOCK_Y TILE_WIDTH

typedef double elemType;

// NbyN X MbyM = NbyM
__global__ void tiledMatMulEigen(Eigen::MatrixXd* m1, Eigen::MatrixXd* m2, Eigen::MatrixXd* m3, int Width)
{
    __shared__ elemType partm1[TILE_WIDTH][TILE_WIDTH];
    __shared__ elemType partm2[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

	// Find x and y indices 
	int row = ty + blockDim.y*by;
	int column = tx + blockDim.x*bx;

    elemType result = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int i = 0; i < int((Width-1)/TILE_WIDTH + 1); ++i) 
    {
        // Collaborative loading of M and N tiles into shared memory
        partm1[ty][tx] = (*m1)(row, i*TILE_WIDTH+tx);
        partm2[ty][tx] = (*m2)(i*TILE_WIDTH+ty, column);
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            result += partm1[ty][j]*partm2[j][tx];
        }
        __syncthreads();
    }
    (*m3)(row,column) = result;
}

// NbyN X MbyM = NbyM
__global__ void tiledMatMul(elemType* m1, elemType* m2, elemType* m3, int Width, int rows1, int cols1, int rows2, int cols2)
{
    __shared__ elemType partm1[TILE_WIDTH][TILE_WIDTH];
    __shared__ elemType partm2[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

	// Find x and y indices 
	int row = ty + blockDim.y*by;
	int column = tx + blockDim.x*bx;

    elemType result = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int i = 0; i < int((Width-1)/TILE_WIDTH + 1); ++i) 
    {
        // Collaborative loading of M and N tiles into shared memory
        if(row < rows1 && i*TILE_WIDTH+tx < cols1) 
        {
            partm1[ty][tx] = m1[row*rows1 + i*TILE_WIDTH+tx];
        }
        else 
        {
            partm1[ty][tx] = 0.0;
        }
        if (i*TILE_WIDTH+ty < rows2 && column < cols2) 
        {
            partm2[ty][tx] = m2[(i*TILE_WIDTH+ty)*rows2 + column];
        }
        else
        {
            partm2[ty][tx] = 0;
        }
        __syncthreads();

        if(row < rows1 && column < cols2)
        {
            for (int j = 0; j < TILE_WIDTH; ++j)
            {
                result += partm1[ty][j]*partm2[j][tx];
            }
        }
        __syncthreads();
    }
    if(row < rows1 && column < cols2)
    {
        m3[row*rows1+column] = result;
    }
}

ViT::ViT(int numLayers, int numPatches, int hiddenSize, int embeddingSize, int numHeads, int numThread, int MLPSize)
{
    this->numThread = numThread;
    this->numLayers = numLayers;
    Wk.resize(numLayers);
    Wq.resize(numLayers);
    Wv.resize(numLayers);
    this->numPatches = numPatches;
    this->hiddenSize = hiddenSize;
    this->embeddingSize = embeddingSize;
    this->numHeads = numHeads;
    this->MLPSize = MLPSize;
    MLPW.resize(numLayers);

    xclass = Eigen::MatrixXd::Random(1, embeddingSize);
    E = Eigen::MatrixXd::Random(embeddingSize, hiddenSize);
    Epos = Eigen::MatrixXd::Random(numPatches+1, hiddenSize);

    streams = new cudaStream_t[numHeads];

    for(int i = 0; i < numLayers; i++)
    {
        // Init MSA
        for(int j = 0; j < numHeads; j++)
        {
            Eigen::MatrixXd wk = Eigen::MatrixXd::Random(hiddenSize, hiddenSize);
            Eigen::MatrixXd wq = Eigen::MatrixXd::Random(hiddenSize, hiddenSize);
            Eigen::MatrixXd wv = Eigen::MatrixXd::Random(hiddenSize, hiddenSize);
            Wk.at(i).push_back(wk);
            Wq.at(i).push_back(wq);
            Wv.at(i).push_back(wv);

            cudaStreamCreate(&streams[j]);
        }   
        
        Eigen::MatrixXd wy = Eigen::MatrixXd::Random(numHeads*hiddenSize, hiddenSize);
        Wy.push_back(wy);  

        // Init MLP
        Eigen::MatrixXd mlpw = Eigen::MatrixXd::Random(hiddenSize, MLPSize);
        Eigen::MatrixXd mlpw2 = Eigen::MatrixXd::Random(MLPSize, hiddenSize);
        MLPW.at(i).push_back(mlpw);
        MLPW.at(i).push_back(mlpw2);

    }
    // Temporarily used msa output
    Y = Eigen::MatrixXd::Zero(numPatches+1, numHeads*hiddenSize);
}

Eigen::MatrixXd ViT::layerNorm(Eigen::MatrixXd* X)
{
    int N = X->rows();
    int M = X->cols();
    Eigen::MatrixXd result = Eigen::MatrixXd::Ones(N, M);
    for(int i = 0; i < N; i++)
    {
        double mean = X->row(i).array().sum()/M;
        result.row(i) = result.row(i)*(-mean);
        Eigen::MatrixXd tmp = X->row(i).array() - mean;
        double std = sqrt(tmp.array().square().sum()/M);
        result.row(i) = (X->row(i) + result.row(i))/std;
    }

    return result;
}

void ViT::softmax(Eigen::MatrixXd* X)
{
    for(int i = 0; i < X->rows(); i++)
    {
        X->block(i,0,1, X->cols()) = X->block(i,0,1, X->cols()).array().exp()/X->block(i,0,1, X->cols()).array().exp().sum();
    }  
}

void ViT::softmaxGPU(Eigen::MatrixXd* X)
{
    for(int i = 0; i < X->rows(); i++)
    {
        X->block(i,0,1, X->cols()) = X->block(i,0,1, X->cols()).array().exp()/X->block(i,0,1, X->cols()).array().exp().sum();
    }  
}

void ViT::softmaxDistributed(Eigen::MatrixXd* X, int rowID)
{
    X->block(rowID,0,1, X->cols()) = X->block(rowID,0,1, X->cols()).array().exp()/X->block(rowID,0,1, X->cols()).array().exp().sum();
}

void ViT::matMulWrapper(Eigen::MatrixXd* m1, Eigen::MatrixXd* m2, Eigen::MatrixXd* m3, int streamID)
{
    // Device pointers
    elemType* d_m1;
    elemType* d_m2;
    elemType* d_m3;
    // Host pointer
    elemType* h_m3;

    // Define sizes
    size_t size1 = m1->size()*sizeof(elemType);
    size_t size2 = m2->size()*sizeof(elemType);
    int newN = m1->rows();
    int newM = m2->cols();
    size_t size3 = newN*newM*sizeof(elemType);

    // Host memory
    h_m3 = (elemType*)malloc(newN*newM*sizeof(elemType));
    // Create GPU memory
    cudaMalloc((void **)&d_m1, size1);
    cudaMalloc((void **)&d_m2, size2);
    cudaMalloc((void **)&d_m3, size3);

    // Copy host memory to GPU
    cudaMemcpyAsync(d_m1, m1->data(), size1, cudaMemcpyHostToDevice, streams[streamID]);
    cudaMemcpyAsync(d_m2, m2->data(), size2, cudaMemcpyHostToDevice, streams[streamID]);
    cudaStreamSynchronize(streams[streamID]);
    // Assign kernel size
    int Width = std::max({m1->rows(), m1->cols(), m2->rows(), m2->cols()});
    dim3 grid = dim3((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 block = dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);

    // Launch kernel
    tiledMatMul<<<grid, block, 0, streams[streamID]>>>(d_m1, d_m2, d_m3, Width, m1->rows(), m1->cols(), m2->rows(), m2->cols());
    cudaStreamSynchronize(streams[streamID]);
    // Copy back to host
    cudaMemcpyAsync(h_m3, d_m3, size3, cudaMemcpyDeviceToHost, streams[streamID]);
    cudaStreamSynchronize(streams[streamID]);
    (*m3) = Eigen::Map<Eigen::MatrixXd>(h_m3, newN, newM);

    // Free memory
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
    free(h_m3);
    cudaStreamSynchronize(streams[streamID]);
}

void ViT::attentionCore(Eigen::MatrixXd* X, int levelID, int headID)
{
    Eigen::MatrixXd K = (*X) * (Wk.at(levelID).at(headID));
    Eigen::MatrixXd Q = (*X) * (Wq.at(levelID).at(headID));
    Eigen::MatrixXd V = (*X) * (Wv.at(levelID).at(headID));
    
    Eigen::MatrixXd dotScores = (Q*K.transpose())/sqrt(hiddenSize);

    // Softmax ------> creates nan or zero values, TO DO
    // Serial softmax
    softmax(&dotScores);
    Eigen::MatrixXd y = dotScores*V;
    Y.block(0, headID*hiddenSize, numPatches+1, hiddenSize).noalias() = y;   
}

void ViT::attentionCoreGPU(Eigen::MatrixXd* X, int levelID, int headID)
{
    Eigen::MatrixXd K;
    matMulWrapper(X, &Wk.at(levelID).at(headID), &K, headID);
    Eigen::MatrixXd Q;
    matMulWrapper(X, &Wq.at(levelID).at(headID), &Q, headID);
    Eigen::MatrixXd V;
    matMulWrapper(X, &Wv.at(levelID).at(headID), &V, headID);

    Eigen::MatrixXd dotScores;
    Eigen::MatrixXd KT = K.transpose();
    matMulWrapper(&Q, &KT, &dotScores, headID);
    dotScores = dotScores/sqrt(hiddenSize);

    // Softmax ------> creates nan or zero values, TO DO
    // Serial softmax
    
    softmax(&dotScores);
    Eigen::MatrixXd y;
    matMulWrapper(&dotScores, &V, &y, headID);
    Y.block(0, headID*hiddenSize, numPatches+1, hiddenSize).noalias() = y; 
}

Eigen::MatrixXd ViT::forward(Eigen::MatrixXd& X)
{
    // Embeddings
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd z0(numPatches+1, hiddenSize);
    z0 << xclass, X*E;
    z0 = z0 + Epos;
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Inference time of the Embeddings : " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

    Eigen::MatrixXd zprev = z0;
    Eigen::MatrixXd zprevLN = z0;
    // std::cout << zprev;
    
    for(int i = 0; i < numLayers; i++)
    {
        // LN
        start = std::chrono::high_resolution_clock::now();
        zprevLN.noalias() = layerNorm(&zprev);
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Layer Normalization 1: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // MSA
        start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < numHeads; j++)
        {            
            threads.push_back(std::thread(&ViT::attentionCore, this, &zprevLN, i, j));
        }

        for (auto& th : threads)
        {
            th.join();
        }
        threads.clear();

        Eigen::MatrixXd zl_hat = Y*(Wy.at(i));

        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " MSA: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // Residual
        start = std::chrono::high_resolution_clock::now();
        zl_hat.noalias() += zprev;
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Residual 1: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // LN
        start = std::chrono::high_resolution_clock::now();
        zprevLN.noalias() = layerNorm(&zl_hat);
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Layer Normalization 2: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // MLP
        start = std::chrono::high_resolution_clock::now();
        zprevLN = zprevLN*MLPW.at(i).at(0);
        zprevLN = zprevLN*MLPW.at(i).at(1);
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " MLP: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // Residual
        start = std::chrono::high_resolution_clock::now();
        zprev.noalias() = zprevLN + zl_hat;
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Residual 2: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;        
    }
    // LN
    start = std::chrono::high_resolution_clock::now();
    zprev = layerNorm(&zprev);
    end = std::chrono::high_resolution_clock::now();
    time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Inference time of the last layer Layer Normalization: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;    

    return zprev;
}

Eigen::MatrixXd ViT::forwardGPU(Eigen::MatrixXd& X)
{
    // Embeddings
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd z0(numPatches+1, hiddenSize);
    Eigen::MatrixXd tempxe;
    matMulWrapper(&X, &E, &tempxe, 0);
    z0 << xclass, tempxe;
    z0 = z0 + Epos;
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Inference time of the Embeddings : " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

    Eigen::MatrixXd zprev = z0;
    Eigen::MatrixXd zprevLN = z0;
    
    for(int i = 0; i < numLayers; i++)
    {
        // LN
        start = std::chrono::high_resolution_clock::now();
        zprevLN.noalias() = layerNorm(&zprev);
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Layer Normalization 1: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // MSA
        start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < numHeads; j++)
        {            
            threads.push_back(std::thread(&ViT::attentionCoreGPU, this, &zprevLN, i, j));
        }
        for (auto& th : threads)
        {
            th.join();
        }
        threads.clear();

        Eigen::MatrixXd zl_hat;
        matMulWrapper(&Y, &(Wy.at(i)), &zl_hat, 0);

        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " MSA: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // Residual
        start = std::chrono::high_resolution_clock::now();
        zl_hat.noalias() += zprev;
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Residual 1: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // LN
        start = std::chrono::high_resolution_clock::now();
        zprevLN.noalias() = layerNorm(&zl_hat);
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Layer Normalization 2: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // MLP
        start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd temp;
        matMulWrapper(&zprevLN, &MLPW.at(i).at(0), &temp, 0);
        matMulWrapper(&temp, &MLPW.at(i).at(1), &zprevLN, 0);
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " MLP: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;

        // Residual
        start = std::chrono::high_resolution_clock::now();
        zprev.noalias() = zprevLN + zl_hat;
        end = std::chrono::high_resolution_clock::now();
        time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference time of the layer " << i << " Residual 2: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;        
    }
    // LN
    start = std::chrono::high_resolution_clock::now();
    zprev = layerNorm(&zprev);
    end = std::chrono::high_resolution_clock::now();
    time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Inference time of the last layer Layer Normalization: " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;    

    return zprev;
}

int main(int argc, char**argv) 
{
    int H = 256;
    int W = 256;
    int C = 3;
    int P = 16;

    int numThread;
    // Patch embeddings -> X->z0;  X=NxP^2.C, z0=(N+1)xD
    int numLayers;      // L
    int numPatches = H*W/pow(P, 2);     // N
    int hiddenSize;     // D
    int embeddingSize = pow(P, 2)*C;  // P^2.C

    // Multihead attention
    int numHeads;    
    int MLPSize;


    if(argc != 1)
    {
        numThread = 1;
        // Patch embeddings -> X->z0;  X=NxP^2.C, z0=(N+1)xD
        numLayers = atoi(argv[1]);      // L
        hiddenSize = atoi(argv[2]);     // D
        MLPSize = atoi(argv[3]);
        numHeads = atoi(argv[4]); 
        std::cout << "Number of layers / Hidden size / MLP Size / Number of head" << std::endl;
        std::cout << numLayers << " " << hiddenSize << " " << MLPSize << " " << numHeads << std::endl;
    }
    else
    {
        numThread = 1;
        // Patch embeddings -> X->z0;  X=NxP^2.C, z0=(N+1)xD
        numLayers = 12;      // L
        hiddenSize = 768; //768;     // D
        MLPSize = 3072;
        numHeads = 12;           
    }

    Eigen::MatrixXd imageFlattened = Eigen::MatrixXd::Random(numPatches, embeddingSize);
    ViT transformer(numLayers, numPatches, hiddenSize, embeddingSize, numHeads, numThread, MLPSize);

    auto start = std::chrono::high_resolution_clock::now();
    // Eigen::MatrixXd result = transformer.forward(imageFlattened);
    Eigen::MatrixXd result = transformer.forwardGPU(imageFlattened);
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Inference time of the ViT Model : " << std::fixed << time_taken/1e6 << std::setprecision(5) << " secs" << std::endl;
    return 0;
}