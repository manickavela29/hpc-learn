#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#define TILE_SIZE 16

#define M 1024
#define N 1024
#define K 1024

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

void init_matrix(float *mat, int rows,int cols) {
    std::random_device dev;
    std::mt19937 rng(128);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,rows*cols);

    for(int i = 0 ; i < rows*cols; i++) {
        mat[i] = static_cast<float>(dist(rng));
    }
}

__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < m && tile * TILE_SIZE + tx < k)
            sharedA[ty][tx] = A[row * k + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if (col < n && tile * TILE_SIZE + ty < k)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        else
            sharedB[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int l = 0; l < TILE_SIZE; ++l)
            sum += sharedA[ty][l] * sharedB[l][tx];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    float *d_a,*d_b,*d_c;

        // Allocate device memory
    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);


    // Kernel launch code
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplyOptimized<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

    // Synchronize device
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}