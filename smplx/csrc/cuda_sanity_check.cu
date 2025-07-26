#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {
    void cuda_vector_add(const float* h_a, const float* h_b, float* h_c, int n) {
        float *d_a, *d_b, *d_c;
        size_t size = n * sizeof(float);
        
        // Allocate device memory
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        
        // Copy input data to device
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        
        // Wait for kernel to complete
        cudaDeviceSynchronize();
        
        // Copy result back to host
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    
    int cuda_device_count() {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }
    
    void cuda_device_info() {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        
        printf("CUDA Device Count: %d\n", device_count);
        
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            printf("Device %d: %s\n", i, prop.name);
            printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
            printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
            printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
            printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        }
    }
}