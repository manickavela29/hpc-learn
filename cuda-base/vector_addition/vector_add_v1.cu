#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

#define N 10000000
#define BLOCK_SIZE 256

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

using namespace std;
// Initilizing vec with random values
void init_vector(float* vec,size_t n) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,n); // distribution in range [1, 6]

    for(int i = 0; i < n; i++) {
        vec[i] = (float)dist(rng);
    }
}

void vector_add_cpu(float *vec_a, float *vec_b, float *vec_c, int size) {
    for(int i = 0; i < size;i++) {
        vec_c[i] = vec_b[i] + vec_b[i];
    }
}

__global__ void vector_add_gpu(float *vec_a, float *vec_b, float *vec_c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        vec_c[i] = vec_a[i] + vec_b[i]; 
    }
}

int main() {

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);


    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    init_vector(h_a,N);
    init_vector(h_b,N);
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);


    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    // Choosing the dims for CUDa kernels with chosen block_size
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Adding Warmp up
    for(int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmarking with average
    double cpu_time = 0.0, gpu_time = 0.0;
    double cpu_total_time = 0, gpu_total_time = 0;

    for(int i=0; i < 20; i++) {
        auto cpu_start = Time::now();
        vector_add_cpu(h_a,h_b,h_c_cpu,N);
        auto cpu_stop = Time::now();
        std::chrono::duration<double,std::milli> cpu_perf = cpu_stop - cpu_start;
        cpu_total_time += cpu_perf.count();
    }
    cpu_time = cpu_total_time / 20.0;

    cout<<"CPU vector add performance : "<< cpu_time <<" ms"<<endl;

    for(int i=0; i < 20; i++) {
        auto gpu_start = Time::now();
        vector_add_gpu<<<num_blocks,BLOCK_SIZE>>>(d_a,d_b,d_c,N);
        cudaDeviceSynchronize();
        auto gpu_stop = Time::now();
        std::chrono::duration<float,std::milli> gpu_perf = gpu_stop - gpu_start;
        gpu_total_time += gpu_perf.count();
    }

    gpu_time = gpu_total_time / 20.0;

    cout<<"GPU vector add performance : "<< gpu_time <<" ms"<<endl;

    // copying data to device
    cudaMemcpy(h_c_gpu,d_c,size,cudaMemcpyDeviceToHost);
    bool verify = true;
    for(int i = 0;i < N; i++) {
        if(fabs(h_c_cpu[i] - h_c_gpu[i])) {
            verify = false;
            break;
        }
    }

    cout<<"Result is " << ( verify? "correct" : "incorrect") <<endl;

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}