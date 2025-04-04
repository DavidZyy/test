// overlap gpu data transfer and kernel execution

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
        data[idx] = data[idx] * 2.0f;
    }
}

void runSameStream(float* h_input, float* h_output, float* d_input, int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Testing kernel and memcpy in same stream...\n");
    CUDA_CHECK(cudaEventRecord(start));
    computeKernel<<<gridSize, blockSize>>>(d_input, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_input, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for same stream: %.2f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void runDifferentStreams(float* h_input, float* h_output, float* d_input, int size, int gridSize, int blockSize) {
    cudaStream_t stream1, stream2;
    cudaEvent_t start, stop, kernelDone;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&kernelDone));
    float milliseconds = 0;

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("\nTesting kernel and memcpy in different streams...\n");
    CUDA_CHECK(cudaEventRecord(start, stream1));
    
    computeKernel<<<gridSize, blockSize, 0, stream1>>>(d_input, size);
    // CUDA_CHECK(cudaEventRecord(kernelDone, stream1));  // 标记kernel完成
    
    // 确保memcpy在kernel完成后执行
    // CUDA_CHECK(cudaStreamWaitEvent(stream2, kernelDone, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_input, size * sizeof(float), cudaMemcpyDeviceToHost, stream2));
    
    CUDA_CHECK(cudaEventRecord(stop, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for different streams: %.2f ms\n", milliseconds);

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(kernelDone));
}

int main() {
    const int SIZE = 1 << 30;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_input = (float*)malloc(SIZE * sizeof(float));
    float *h_output = (float*)malloc(SIZE * sizeof(float));
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = (float)i;
    }

    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(float)));

    // 预热GPU，避免初始化开销影响测量
    computeKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 运行测试
    runSameStream(h_input, h_output, d_input, SIZE, GRID_SIZE, BLOCK_SIZE);
    runDifferentStreams(h_input, h_output, d_input, SIZE, GRID_SIZE, BLOCK_SIZE);
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_output[i] != h_input[i] * 2.0f) {
            correct = false;
            break;
        }
    }
    printf("Same stream verification: %s\n", correct ? "PASS" : "FAIL");

    runSameStream(h_input, h_output, d_input, SIZE, GRID_SIZE, BLOCK_SIZE);
    runDifferentStreams(h_input, h_output, d_input, SIZE, GRID_SIZE, BLOCK_SIZE);
    
    // 验证结果
    correct = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_output[i] != h_input[i] * 2.0f) {
            correct = false;
            break;
        }
    }
    printf("Different streams verification: %s\n", correct ? "PASS" : "FAIL");

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    free(h_input);
    free(h_output);

    return 0;
}
