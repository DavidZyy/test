#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cblas.h>  // Include OpenBLAS

#define MEASURE_TIME(FUNC, DESC) \
    start = std::chrono::high_resolution_clock::now(); \
    FUNC; \
    end = std::chrono::high_resolution_clock::now(); \
    duration = end - start; \
    std::cout << DESC << " Time: " << duration.count() << " seconds" << std::endl;

void generate_matrices(float* A, float* B, int size) {
    for (int i = 0; i < size * size; ++i) {
        // A[i] = static_cast<float>(rand()) / RAND_MAX;
        // B[i] = static_cast<float>(rand()) / RAND_MAX;
        A[i] = 1;
        B[i] = 1;
    }
}

bool compare_results(float* C1, float* C2, int size, float tol = 1e-3) {
    for (int i = 0; i < size * size; ++i) {
        if (fabs(C1[i] - C2[i]) > tol) {
            std::cout << C1[i] << " != " << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

/////////////////////////// gemm methods //////////////////////////////
void gemm_cuda(float* A, float* B, float* C, int M, int N, int K);
void gemm_cublas(float* A, float* B, float* C, int M, int N, int K);

void gemm_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0.0f;
            for (int e = 0; e < K; ++e) {
                value += A[i * K + e] * B[e * N + j];
            }
            C[i * N + j] = value;
        }
    }
}

void gemm_cblas(float* A, float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
///////////////////////////////////////////////////////////////////////

void benchmark_gemm(int size) {
    float *A = new float[size * size];
    float *B = new float[size * size];
    float *C_cpu = new float[size * size];
    float *C_cblas = new float[size * size];
    float *C_cuda = new float[size * size];
    float *C_cublas = new float[size * size];

    generate_matrices(A, B, size);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration;

    // MEASURE_TIME(gemm_cpu(A, B, C_cpu, size, size, size), "CPU GEMM");
    MEASURE_TIME(gemm_cblas(A, B, C_cblas, size, size, size), "cBLAS GEMM");
    // MEASURE_TIME(gemm_cuda(A, B, C_cuda, size, size, size), "CUDA GEMM");
    MEASURE_TIME(gemm_cublas(A, B, C_cublas, size, size, size), "cuBLAS GEMM");

    if (compare_results(C_cpu, C_cblas, size)) {
        // if (compare_results(C_cpu, C_cublas, size)) {
            std::cout << "Results are correct and match." << std::endl;
        // } else {
        //     std::cout << "Results are correct but do not match cuBLAS." << std::endl;
        // }
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_cblas;
    delete[] C_cuda;
    delete[] C_cublas;
}

int main() {
    // int size = 32;  // Example size, you can vary this
    // int size = 1024;
    // int size = 2048;
    // int size = 4096;
    // int size = 8192;
    int size = 8192 * 2;
    benchmark_gemm(size);
    return 0;
}
