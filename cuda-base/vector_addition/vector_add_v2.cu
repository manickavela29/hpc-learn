#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

using namespace std;

// Initilizing vec with random values
void init_vector(float* vec,size_t n) {
    std::random_device dev;
    std::mt19937 rng(128);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,n); // distribution in range [1, 6]

    for(int i = 0; i < n; i++) {
        vec[i] = (float)dist(rng);
    }
}

void vector_add_cpu(float *vec_a, float *vec_b, float *vec_c, int size) {
    for(int i = 0; i < size; i++) {
        vec_c[i] = vec_a[i] + vec_b[i];
    }
}

__global__ void vector_add_gpu_1d(float *vec_a, float *vec_b, float *vec_c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        vec_c[i] = vec_a[i] + vec_b[i]; 
    }
}

__global__ void vector_add_gpu_3d(float *vec_a, float *vec_b, float *vec_c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + (j * nx) + (k * nx * ny);
        if (idx < nx * ny * nz) {
            vec_c[idx] = vec_a[idx] + vec_b[idx];
        }
    }
}

int main() {

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    init_vector(h_a,N);
    init_vector(h_b,N);
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c_1d,size);
    cudaMalloc(&d_c_3d,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Choosing the dims for CUDa kernels with chosen block_size
    int num_blocks = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // grid and block size dims for 3D
    int nx = 100000, ny = 100, nz = 1;
    dim3 block_size_3d(BLOCK_SIZE_3D_X,BLOCK_SIZE_3D_Y,BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    cout << "Running warmup..." << endl;
    // Adding Warmp up
    for(int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<num_blocks, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_blocks, block_size_3d>>>(d_a,d_b,d_c_3d,nx,ny,nz);
        cudaDeviceSynchronize();
    }

    // Benchmarking with average
    double cpu_time = 0.0, gpu_time_1d = 0.0, gpu_time_3d = 0.0;
    double cpu_total_time = 0, gpu_total_time_1d = 0, gpu_total_time_3d = 0;;

    for(int i=0; i < 20; i++) {
        auto cpu_start = Time::now();
        vector_add_cpu(h_a,h_b,h_c_cpu,N);
        auto cpu_stop = Time::now();
        std::chrono::duration<double,std::milli> cpu_perf = cpu_stop - cpu_start;
        cpu_total_time += cpu_perf.count();
    }
    cpu_time = cpu_total_time / 20.0;

    for(int i=0; i < 20; i++) {
        cudaMemset(d_c_1d, 0, size);  // Clear previous results
        auto gpu_start = Time::now();
        vector_add_gpu_1d<<<num_blocks,BLOCK_SIZE_1D>>>(d_a,d_b,d_c_1d,N);
        cudaDeviceSynchronize();
        auto gpu_stop = Time::now();
        std::chrono::duration<float,std::milli> gpu_perf = gpu_stop - gpu_start;
        gpu_total_time_1d += gpu_perf.count();
    }
    gpu_time_1d = gpu_total_time_1d / 20.0;

    for(int i=0; i < 20; i++) {
        cudaMemset(d_c_3d, 0, size);  // Clear previous results
        auto gpu_start = Time::now();
        vector_add_gpu_3d<<<num_blocks_3d,block_size_3d>>>(d_a,d_b,d_c_3d,nx,ny,nz);
        cudaDeviceSynchronize();
        auto gpu_stop = Time::now();
        std::chrono::duration<float,std::milli> gpu_perf = gpu_stop - gpu_start;
        gpu_total_time_3d += gpu_perf.count();
    }
    gpu_time_3d = gpu_total_time_3d / 20.0;

    std::cout<<"CPU vector add performance        : "<< cpu_time <<" ms"<<endl;
    cout<<"GPU vector add performance        : "<< gpu_time_1d <<" ms"<<endl;
    cout<<"GPU vector add blocks performance : "<< gpu_time_3d <<" ms\n"<<endl;

    cout<<"Speedup CPU vs GPU 1D    " << (cpu_time/gpu_time_1d) <<endl;
    cout<<"Speedup CPU vs GPU 3D    " << (cpu_time/gpu_time_3d) <<endl;
    cout<<"Speedup GPU 1D va GPU 3D " << (gpu_time_1d/gpu_time_3d) << "\n" <<endl;


    cout << "Verifying accuracy..." << endl;
    // copying data to device
    cudaMemcpy(h_c_gpu_1d,d_c_1d,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_gpu_3d,d_c_3d,size,cudaMemcpyDeviceToHost);
    bool verify1 = true, verify2 = true;
    for(int i = 0; i < N; i++) {
        if(fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-5) {
            verify1 = false;
            break;
        }
    }

    for(int i = 0; i < N; i++) {
        if(fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-5) {
            verify2 = false;
            break;
        }
    }

    cout<<"Result GPU vector additiona is " << ( verify1 ? "correct" : "incorrect") <<endl;
    cout<<"Result GPU block vector additiona is " << ( verify2 ? "correct" : "incorrect") <<endl;

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}