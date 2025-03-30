#include <cuda_runtime.h>
#include <stdio.h>

__global__ void multiply(float *x, float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * y[i];
}

__global__ void add(float *a, float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// 不使用流的情况
float no_stream(float *x, float *y, float *w, float *z, float *temp, int n, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    multiply<<<grid, block>>>(x, y, temp, n);  // 默认流
    add<<<grid, block>>>(temp, w, z, n);       // 默认流
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds / 1000.0f;  // 转换为秒
}

// 使用流的情况
float with_streams(float *x, float *y, float *w, float *z, float *temp, int n, dim3 grid, dim3 block) {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    multiply<<<grid, block, 0, stream1>>>(x, y, temp, n);  // stream1
    add<<<grid, block, 0, stream2>>>(temp, w, z, n);       // stream2
    cudaEventRecord(stop, 0);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds / 1000.0f;  // 转换为秒
}

int main() {
    int n = 1 << 20;  // 1M 元素
    float *x, *y, *w, *z, *temp;

    // 分配显存
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    cudaMalloc(&w, n * sizeof(float));
    cudaMalloc(&z, n * sizeof(float));
    cudaMalloc(&temp, n * sizeof(float));

    // 初始化数据（简单填充）
    float *h_x = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
    }
    cudaMemcpy(x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h_x);

    // 设置线程块和网格
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // 测试多次取平均值
    int num_runs = 10;
    float no_stream_time = 0.0f;
    float with_streams_time = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        no_stream_time += no_stream(x, y, w, z, temp, n, grid, block);
        with_streams_time += with_streams(x, y, w, z, temp, n, grid, block);
    }
    no_stream_time /= num_runs;
    with_streams_time /= num_runs;

    // 输出结果
    printf("不使用CUDA流平均时间: %.6f 秒\n", no_stream_time);
    printf("使用CUDA流平均时间: %.6f 秒\n", with_streams_time);
    printf("时间差异: %.6f 秒\n", no_stream_time - with_streams_time);

    // 清理
    cudaFree(x);
    cudaFree(y);
    cudaFree(w);
    cudaFree(z);
    cudaFree(temp);

    return 0;
}
