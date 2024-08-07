// includes
#include <curand_mtgp32_kernel.h>
#include <stdio.h>
#include "cuda_runtime_api.h"

// CUDA kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    // for (int i = index; i < n; i += stride) {
    //     y[i] = x[i] + y[i];
    // }

    // 另一个版本
    // index可以索引到所有的线程，让一个线程只处理一个元素
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}

int main(void) {
    int N = 1<<20; // 1M elements
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    printf("Max error: %f\n", maxError);

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
