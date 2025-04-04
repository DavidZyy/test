#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void computeKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1.0f;
    }
}

int times = 10;

__global__ void compareKernel(int* data, int times, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] != times) {
            printf("Mismatch at index %d: %d != %d\n", idx, data[idx], times);
        }
    }
}

void kernel_time(int *buffer, int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < times; i++) {
        computeKernel<<<gridSize, blockSize>>>(buffer, size);
    }

    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize()); // no need

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for kernel execution: %.2f ms\n", milliseconds);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    compareKernel<<<gridSize, blockSize>>>(buffer, times, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void copy_time(int *dst, int *src,  int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(dst, src, size* sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize()); no need 
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for memory copy: %.2f ms\n", milliseconds);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    compareKernel<<<gridSize, blockSize>>>(dst, times, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish, or you might not see the output in compareKernel
}

void runSameStream(int *buffer, int *dst, int *src, int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;
    int times = 10;

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(dst, src, size* sizeof(int), cudaMemcpyHostToDevice));
    for (int i = 0; i < times; i++) {
        computeKernel<<<gridSize, blockSize>>>(buffer, size);
    }
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize()); // no need stop will wait kernel to finish

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for same stream: %.2f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    compareKernel<<<gridSize, blockSize>>>(buffer, times, size);
    compareKernel<<<gridSize, blockSize>>>(dst, times, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runDifferentStreams(int *buffer, int *dst, int *src, int size, int gridSize, int blockSize) {
    cudaStream_t stream1, stream2;
    cudaEvent_t start1, stop1, start2, stop2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));
    float milliseconds = 0;
    int times = 10;

    CUDA_CHECK(cudaEventRecord(start1, stream1));
    CUDA_CHECK(cudaEventRecord(start2, stream2));

    // CUDA_CHECK(cudaMemcpyAsync(dst, src, size * sizeof(int), cudaMemcpyHostToDevice, stream2));  // put here have no overlap ...
    for (int i = 0; i < times; i++) {
        computeKernel<<<gridSize, blockSize,0, stream1>>>(buffer, size);
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size * sizeof(int), cudaMemcpyHostToDevice, stream2)); // put here have overlap ...

    // CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop1, stream1));
    CUDA_CHECK(cudaEventRecord(stop2, stream2));
    // CUDA_CHECK(cudaStreamSynchronize(stream1));
    // CUDA_CHECK(cudaStreamSynchronize(stream2));
    CUDA_CHECK(cudaDeviceSynchronize());

    // find the max time of start1 - stop1 and start2 - stop2, start1 - stop2 and start2 - stop1
    // ...
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start1, stop2));
    printf("Time for different streams: %.2f ms\n", milliseconds);

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));

    compareKernel<<<gridSize, blockSize>>>(buffer, times, size);
    compareKernel<<<gridSize, blockSize>>>(dst, times, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    const int SIZE = 1 << 26;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_buffer, *d_buffer1, *d_buffer2;

    h_buffer = (int*)malloc(SIZE * sizeof(int));
    CUDA_CHECK(cudaMalloc((void**)&d_buffer1, SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_buffer2, SIZE * sizeof(int)));

    for (int i = 0; i < SIZE; i++) {
        h_buffer[i] = times;
    }

    CUDA_CHECK(cudaMemset(d_buffer1, 0, SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buffer2, 0, SIZE * sizeof(int)));

    runDifferentStreams(d_buffer1, d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    CUDA_CHECK(cudaMemset(d_buffer1, 0, SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buffer2, 0, SIZE * sizeof(int)));

    kernel_time(d_buffer1, SIZE, GRID_SIZE, BLOCK_SIZE);
    copy_time(d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    CUDA_CHECK(cudaMemset(d_buffer1, 0, SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buffer2, 0, SIZE * sizeof(int)));

    runSameStream(d_buffer1, d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    CUDA_CHECK(cudaMemset(d_buffer1, 0, SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buffer2, 0, SIZE * sizeof(int)));
}
