#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

using namespace std;

#define M 256
#define N 256
#define K 256
#define BLOCK_SIZE 256 

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

void init_matrix(float *mat, int rows,int cols) {
    std::random_device dev;
    std::mt19937 rng(128);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,rows*cols);

    for(int i = 0 ; i < rows*cols; i++) {
        mat[i] = (float)dist(rng);
    }
}

void matmul_cpu(float *A,float *B, float *C,int m,int n,int k) {
    for (int i = 0; i < m;i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++ ) {
                sum += A[n * i + l] * B[k * l + j];
             }
            C[i * m + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A,float *B, float *C,int m,int n,int k) {
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k) {
        float sum = 0.0f;
        for(int l = 0; l < n; l++) {
            sum += A[row * k + l] * B[k * l +  col];
        }
        C[row * n + col] = sum;
    }

}

int main() {

    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;

    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);
    
    init_matrix(h_a,M,N);
    init_matrix(h_b,N,K);

    cudaMalloc(&d_a,size_A);
    cudaMalloc(&d_b,size_B);
    cudaMalloc(&d_c,size_C);

    cudaMemcpy(d_a,h_a,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_B,cudaMemcpyDeviceToHost);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm up
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        cudaDeviceSynchronize();
    }

   double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        auto cpu_start = Time::now();
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        auto cpu_stop = Time::now();
        std::chrono::duration<double,std::milli> cpu_perf = cpu_stop - cpu_start;
        cpu_total_time += cpu_perf.count();
    }
    double cpu_avg_time = cpu_total_time / 20.0;


    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        auto gpu_start = Time::now();
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
        auto gpu_stop = Time::now();
        std::chrono::duration<float,std::milli> gpu_perf = gpu_stop - gpu_start;
        gpu_total_time += gpu_perf.count();
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));

        // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}