#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <iostream>
#include <vector>
#include <random>
#include <functional>

#define M 4096
#define K 1024
#define N 4096


using namespace std;

// Naive CUDA kernel for matrix multiplication
__global__ void naiveMatrixMultiply(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(float *mat, int rows,int cols) {
    std::random_device dev;
    std::mt19937 rng(128);
    std::uniform_real_distribution<float> dist(-0.5,0.5);

    for(int i = 0 ; i < rows*cols; i++) {
        mat[i] = static_cast<float>(dist(rng));
    }
}

float time_kernel(std::function<void()> kernel_func) {
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_time;
}

float benchmark_kernel(std::function<void()> kernel_func,int warmup, int benchrun) {

    //Warm up
    for(int i = 0; i < warmup; i++) {
        kernel_func();
    }

    vector<float> times;
    //benchamrk
    for(int i = 0; i < benchrun; i++) {
        float time = time_kernel(kernel_func);
        times.push_back(time);
    }

    float avg_time = std::accumulate(times.begin(),times.end(),0.0f) / benchrun;
    return avg_time;
}

bool verifyResults(const vector<float>& expected, const vector<float>& actual, float tolerance = 1e-2) {
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        float rel_error = std::abs(expected[i] - actual[i]);
        if (rel_error > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << actual[i] << ", relative error: " << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

int main() {

    vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    vector<float> h_C_cublas_fp32(M * N), h_C_cublasLt_fp32(M * N);
    vector<float> h_C_cublas_fp16(M * N), h_C_cublasLt_fp16(M * N);
    vector<float> h_C_naive(M * N);
    vector<half> h_C_half(M * N);

    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);

    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half, *d_C_half;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_A_half, M * K * sizeof(half));
    cudaMalloc(&d_B_half, K * N * sizeof(half));
    cudaMalloc(&d_C_half, M * N * sizeof(half));

    cudaMemcpy(d_A,h_A.data(), M * K * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B.data(), K * N * sizeof(float),cudaMemcpyHostToDevice);

    // Handling Half precision
    vector<half> h_A_half(M * K), h_B_half(K * N);
    for(int i = 0; i < M * K; i++) h_A_half[i] = __float2half(h_A[i]);
    for(int i = 0; i < K * N; i++) h_B_half[i] = __float2half(h_B[i]);

    cudaMemcpy(d_A_half, h_A_half.data(), M * K * sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B_half.data(), K * N * sizeof(half),cudaMemcpyHostToDevice);

    const int warmup = 3, benchrun = 10;

    //Benchmarking Native CUDA Matmul

    dim3 blockDim(32,32);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y -1)/blockDim.y);

    float naive_cuda_time = benchmark_kernel([&]() {
        naiveMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }, warmup, benchrun);

    cudaMemcpy(h_C_naive.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Benchamrking with cublas
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublasLt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublasLt_handle);

    float alpha = 1.0f, beta = 0.0f;
    half alpha_half = __float2half(1.0), beta_half = __float2half(0.0f);

    float cublas_fp32_time = benchmark_kernel([&]() {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }, warmup, benchrun);

    float cublas_fp16_time = benchmark_kernel([&]() {
        cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_half, d_B_half, N, d_A_half, K, &beta_half, d_C_half, N);
    }, warmup, benchrun);
    cudaMemcpy(h_C_half.data(), d_C_half, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < M * N; i++) h_C_cublas_fp16[i] = __half2float(h_C_half[i]);

    cudaMemcpy(h_C_cublas_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_C_cublas_fp16.data(), d_C_half, K * N * sizeof(half), cudaMemcpyDeviceToHost);

    // Becnhamrkaing with cublasLt
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulDescCreate(&operationDesc,CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N);

    cublasLtMatmulDesc_t operationDesc_half = nullptr;
    cublasLtMatrixLayout_t Adesc_half = nullptr, Bdesc_half = nullptr, Cdesc_half = nullptr;
    cublasLtMatmulDescCreate(&operationDesc_half, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    cublasLtMatrixLayoutCreate(&Adesc_half, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&Bdesc_half, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&Cdesc_half, CUDA_R_16F, N, M, N);

    float cublasLt_fp32_time = benchmark_kernel([&]() {
        cublasLtMatmul(cublasLt_handle, operationDesc, &alpha, d_B, Bdesc, d_A, Adesc, &beta, d_C, Cdesc, d_C, Cdesc, nullptr, nullptr, 0, 0);
    }, warmup, benchrun);

    float cublasLt_fp16_time = benchmark_kernel([&]() {
        cublasLtMatmul(cublasLt_handle, operationDesc_half, &alpha_half, d_B_half, Bdesc_half, d_A_half, Adesc_half, &beta_half, d_C_half, Cdesc_half, d_C_half, Cdesc_half, nullptr, nullptr, 0, 0);
    }, warmup, benchrun);

    cudaMemcpy(h_C_cublasLt_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_half.data(), d_C_half, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < M * N; i++) h_C_cublasLt_fp16[i] = __half2float(h_C_half[i]);

    float max_error_fp16_cublas = 0.0f;
    float max_error_fp16_cublasLt = 0.0f;
    float max_error_fp32_cublas = 0.0f;
    float max_error_fp32_cublasLt = 0.0f;

    for(int i = 0 ; i < M * N ; i++) {
        float error = std::abs(h_C_naive[i] - h_C_cublas_fp16[i]);
        if(error > max_error_fp16_cublas) {
            max_error_fp16_cublas = error;
        }

        error = std::abs(h_C_naive[i] - h_C_cublasLt_fp16[i]);
        if(error > max_error_fp16_cublasLt) {
            max_error_fp16_cublasLt = error;
        }

        error = std::abs(h_C_naive[i] - h_C_cublas_fp32[i]);
        if(error > max_error_fp32_cublas) {
            max_error_fp32_cublas = error;
        }

        error = std::abs(h_C_naive[i] - h_C_cublasLt_fp32[i]);
        if(error > max_error_fp32_cublasLt) {
            max_error_fp32_cublasLt = error;
        }
    }


    cout <<"\nPerformance Testing" << endl;
    cout <<"================\n" << endl;

    cout <<"Naive CUDA matmul    : " << naive_cuda_time << endl;
    cout <<"cuBlAS FP32 matmul   : " << cublas_fp32_time << endl;
    cout <<"cuBlASLt FP32 matmul : " << cublasLt_fp32_time << endl;
    cout <<"cuBlAS FP16 matmul   : " << cublas_fp16_time << endl;
    cout <<"cuBlASLt FP16 matmul : " << cublasLt_fp16_time << endl;

    cout <<"\n\nAccuracy Testing" << endl;
    cout <<"================\n" << endl;

    cout << "cuBLAS max fp16 error : " << max_error_fp16_cublas << endl;
    cout << "cuBLASLT max fp16 error : " << max_error_fp16_cublasLt << endl;
    cout << "cuBLAS max fp32 error : " << max_error_fp32_cublas << endl;
    cout << "cuBLASLT max fp32 error : " << max_error_fp32_cublasLt << endl;

    bool cublas_fp32_correct = verifyResults(h_C_naive,h_C_cublas_fp32, 1e-2);
    bool cublasLt_fp32_correct = verifyResults(h_C_naive, h_C_cublasLt_fp32, 1e-2);
    bool cublas_fp16_correct = verifyResults(h_C_naive, h_C_cublas_fp16, 5e-1);
    bool cublasLt_fp16_correct = verifyResults(h_C_naive, h_C_cublasLt_fp16, 5e-1);

    std::cout << "cuBLAS FP32 results " << (cublas_fp32_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 1e-2." << std::endl;
    std::cout << "cuBLASLt FP32 results " << (cublasLt_fp32_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 1e-2." << std::endl;
    std::cout << "cuBLAS FP16 results " << (cublas_fp16_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 5e-1." << std::endl;
    std::cout << "cuBLASLt FP16 results " << (cublasLt_fp16_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 5e-1." << std::endl;

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc_half);
    cublasLtMatrixLayoutDestroy(Adesc_half);
    cublasLtMatrixLayoutDestroy(Bdesc_half);
    cublasLtMatrixLayoutDestroy(Cdesc_half);
    cublasLtDestroy(cublasLt_handle);
    cublasDestroy(cublas_handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C_half);

    return 0;
}

