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

__global__ void computeKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1.0f;
    }
}

void kernel_time(float *buffer, int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;
    int times = 10;

    CUDA_CHECK(cudaEventRecord(start));
    // launch times kernel(for example, 100 launches overlap one memcpy)
    for (int i = 0; i < times; i++) {
        computeKernel<<<gridSize, blockSize>>>(buffer, size);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for kernel execution: %.2f ms\n", milliseconds);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void copy_time(float *dst, float *src,  int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(dst, src, size* sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for memory copy: %.2f ms\n", milliseconds);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void copy_time_async(float *dst, float *src,  int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size* sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for memory copy async: %.2f ms\n", milliseconds);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void runSameStream(float *buffer, float *dst, float *src, int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;
    int times = 10;

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(dst, src, size* sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < times; i++) {
        computeKernel<<<gridSize, blockSize>>>(buffer, size);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for same stream: %.2f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void runDifferentStreams(float *buffer, float *dst, float *src, int size, int gridSize, int blockSize) {
    cudaStream_t stream1, stream2;
    cudaEvent_t start, stop, kernelDone;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // CUDA_CHECK(cudaEventCreate(&kernelDone));
    float milliseconds = 0;
    int times = 10;

    CUDA_CHECK(cudaEventRecord(start, stream1));

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size * sizeof(float), cudaMemcpyHostToDevice, stream2));
    for (int i = 0; i < times; i++) {
        computeKernel<<<gridSize, blockSize,0, stream1>>>(buffer, size);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for different streams: %.2f ms\n", milliseconds);

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    const int SIZE = 1 << 26;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_buffer = (float*)malloc(SIZE * sizeof(float));
    for (int i = 0; i < SIZE; i++) {
        h_buffer[i] = (float)1.0f;
    }
    
    float* d_buffer1, *d_buffer2;
    CUDA_CHECK(cudaMalloc((void**)&d_buffer1, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_buffer2, SIZE * sizeof(float)));

    // copy h_buffer to d_buffer1
    CUDA_CHECK(cudaMemcpy(d_buffer1, h_buffer, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // warm up
    for (int i = 0; i < 10; i++) {
        computeKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_buffer1, SIZE);
    }

    runDifferentStreams(d_buffer1, d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    runSameStream(d_buffer1, d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    copy_time(d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    // copy_time_async(d_buffer2, h_buffer, SIZE, GRID_SIZE, BLOCK_SIZE);
    kernel_time(d_buffer1, SIZE, GRID_SIZE, BLOCK_SIZE);
}
