#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

using namespace std;

#define M 3
#define K 4
#define N 2

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {

    float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float B[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float C_cpu[M * N], C_cublas_s[M * N], C_cublas_h[M * N];

    // CPU matmul call
    cpu_matmul(A,B,C_cpu);

    // Cublas matmul 

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,M * K * sizeof(float));
    cudaMalloc(&d_B,K * N * sizeof(float));
    cudaMalloc(&d_C,M * N * sizeof(float));

    cudaMemcpy(A,d_A,M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B,d_B,K * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    // Warmp up for sgemm
    for(int i = 0; i < 3; i++) {
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,d_B,N,d_A,K,&beta,d_C,N);
    }

    auto sgemm_total_time = 0.0;
    for(int i = 0; i < 10; i++) {
        auto sgemm_start_time = Time::now();
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,d_B,N,d_A,K,&beta,d_C,N);
        auto sgemm_stop_time = Time::now();
        std::chrono::duration<double,std::milli> sgemm_perf = sgemm_stop_time - sgemm_start_time;
        sgemm_total_time += sgemm_perf.count();
    }
    cout << "SGEMM timing : " << (sgemm_total_time/10.0)  * 1e6f <<endl;

    // cuBlas sgemm
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,d_B,N,d_A,K,&beta,d_C,N);
    cudaMemcpy(C_cublas_s,d_C,M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // cuBlas hgemm
    // Must convert float to half dtype
    half *d_A_h, *d_B_h, *d_C_h;
    cudaMalloc(&d_A_h, M * K * sizeof(half));
    cudaMalloc(&d_B_h, K * N * sizeof(half));
    cudaMalloc(&d_C_h, M * N * sizeof(half));

    half A_h[M * K], B_h[K * N];
    for(int i = 0; i < M * K; i++) {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = __float2half(B[i]);
    }

    cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice);

    __half alpha_h = __float2half(1.0), beta_h = __float2half(0.0f);

    // Warmp up for hgemm
    for(int i = 0; i < 3; i++) {
        cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K,&alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N);
    }

    double hgemm_total_time = 0.0;
    for(int i = 0; i < 10; i++) {
        auto hgemm_start_time = Time::now();
        cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K,&alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N);
        auto hgemm_stop_time = Time::now();
        std::chrono::duration<float,std::milli> hgemm_perf = hgemm_stop_time - hgemm_start_time;
        hgemm_total_time += hgemm_perf.count();
    }
    cout << "HGEMM timing : " << (hgemm_total_time/10.0) * 1e6f << endl;

    half C_h[M * N];
    cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_h);
    cudaFree(d_B_h);
    cudaFree(d_C_h);
    cublasDestroy(handle);

    return 0;
}