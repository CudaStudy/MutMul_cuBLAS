
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "DS_timer.cuh"

#define ROW_SIZE (32)
#define K_SIZE (32)
#define COL_SIZE (32)
#define WORK_LOAD (ROW_SIZE * COL_SIZE)

#define INDEX2ROW(_index, _width) (int)((_index) / (_width))
#define INDEX2COL(_index, _width) ((_index) % (_width))
#define ID2INDEX(_row, _col, _width) (((_row) * (_width)) + (_col))

#define BLOCK_SIZE 16

#define CPU1 0
#define CPU2 1
#define GPU1 2
#define GPU2 3
#define GPU3 4

__global__ void simple_matmul(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= _m || col >= _n)
        return;

    float val = 0;
    for (int i = 0; i < _k; i++)
    {
        val += __fmul_rn(_matA[ID2INDEX(row, i, _k)], _matB[ID2INDEX(i, col, _n)]);
    }
    _matC[ID2INDEX(row, col, _n)] = val;
}

__global__ void shared_matmul(float *_A, float *_B, float *_C, int _m, int _n, int _k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    int row_offset = threadIdx.x;
    int col_offset = threadIdx.y;

    float val = 0;
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE]; // 16 * 16 * 8B = 2048B
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE]; // 16 * 16 * 8B = 2048B

    for (int i = 0; i < ceil((float)_k / BLOCK_SIZE); i++)
    {
        int block_offset = i * BLOCK_SIZE;

        if (row >= _m || block_offset + col_offset >= _k)
        {
            sA[row_offset][col_offset] = 0;
        }
        else
        {
            sA[row_offset][col_offset] = _A[ID2INDEX(row, block_offset + col_offset, _k)];
        }
        if (col >= _n || block_offset + row_offset >= _k)
        {
            sB[row_offset][col_offset] = 0;
        }
        else
        {
            sB[row_offset][col_offset] = _B[ID2INDEX(block_offset + row_offset, col, _n)];
        }
        __syncthreads(); // wait until all thread load the matrix

        // matrix multiplication
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            val += __fmul_rn(sA[row_offset][i], sB[i][col_offset]);
        }
        __syncthreads(); // wait until all thread load the matrix
    }

    if (row >= _m || col >= _n)
        return;

    _C[ID2INDEX(row, col, _n)] = val;
}

int main(int argc, char *argv[])
{
    printf("========================Application Start========================\n");

    DS_timer timer(5);
    timer.setTimerName(CPU1, (char *)"CPU_row_major");
    timer.setTimerName(CPU2, (char *)"CPU_col_major");
    timer.setTimerName(GPU1, (char *)"GPU_Simple");
    timer.setTimerName(GPU2, (char *)"GPU_Shared");
    timer.setTimerName(GPU3, (char *)"GPU_cuBLAS");

    int m,
        n, k;
    if (argc < 3)
    {
        m = ROW_SIZE;
        n = COL_SIZE;
        k = K_SIZE;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    printf("Matrix Size\n");
    printf("A : (%d, %d), B : (%d, %d), C : (%d, %d)\n", m, k, k, n, m, n);

    long unsigned matA_size = m * k;
    long unsigned matB_size = k * n;
    long unsigned matC_size = m * n;

    // host memroy allocation & initialization
    float *H1_A = new float[matA_size]; // row major
    float *H1_B = new float[matB_size]; // row major
    float *H1_C = new float[matC_size]; // row major

    float *H2_A = new float[matA_size]; // col major
    float *H2_B = new float[matB_size]; // col major
    float *H2_C = new float[matC_size]; // col major

    float *G1_C = new float[matC_size]; // simple matmul
    float *G2_C = new float[matC_size]; // shared tiling
    float *G3_C = new float[matC_size]; // cublas segmm

    memset(H1_A, 0, sizeof(float) * matA_size);
    memset(H1_B, 0, sizeof(float) * matB_size);
    memset(H1_C, 0, sizeof(float) * matC_size);
    memset(G1_C, 0, sizeof(float) * matC_size);
    memset(G2_C, 0, sizeof(float) * matC_size);
    memset(G3_C, 0, sizeof(float) * matC_size);

    // generate rand matrix
    for (int i = 0; i < matA_size; i++)
    {
        H1_A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
        H2_A[i] = H1_A[i];
    }
    for (int i = 0; i < matB_size; i++)
    {
        H1_B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
        H2_B[i] = H1_B[i];
    }

    // CPU row major Matrix Multiplication
    timer.onTimer(CPU1);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            int idx = ID2INDEX(row, col, n);
            H1_C[idx] = 0;
            for (int i = 0; i < k; i++)
            {
                H1_C[idx] += H1_A[ID2INDEX(row, i, k)] * H1_B[ID2INDEX(i, col, n)];
            }
        }
    }
    timer.offTimer(CPU1);

    // CPU col major Matrix Multiplication
    timer.onTimer(CPU2);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            int idx = ID2INDEX(row, col, n);
            H2_C[idx] = 0;
            for (int i = 0; i < k; i++)
            {
                H2_C[idx] += H2_A[ID2INDEX(row, i, k)] * H2_B[ID2INDEX(i, col, n)];
            }
        }
    }
    timer.offTimer(CPU2);

    // device memory allocation & initialization
    // 1: simple, 2: shared, 3: cublas
    float *D1_A, *D1_B, *D1_C;
    D1_A = D1_B = D1_C = NULL;

    float *D2_A, *D2_B, *D2_C;
    D2_A = D2_B = D2_C = NULL;

    float *D3_A, *D3_B, *D3_C;
    D3_A = D3_B = D3_C = NULL;

    cudaMalloc(&D1_A, sizeof(float) * matA_size);
    cudaMalloc(&D1_B, sizeof(float) * matB_size);
    cudaMalloc(&D1_C, sizeof(float) * matC_size);

    cudaMalloc(&D2_A, sizeof(float) * matA_size);
    cudaMalloc(&D2_B, sizeof(float) * matB_size);
    cudaMalloc(&D2_C, sizeof(float) * matC_size);

    cudaMalloc(&D3_A, sizeof(float) * matA_size);
    cudaMalloc(&D3_B, sizeof(float) * matB_size);
    cudaMalloc(&D3_C, sizeof(float) * matC_size);

    cudaMemset(&D1_A, 0, sizeof(float) * matA_size);
    cudaMemset(&D1_B, 0, sizeof(float) * matB_size);
    cudaMemset(&D1_C, 0, sizeof(float) * matC_size);

    cudaMemset(&D2_A, 0, sizeof(float) * matA_size);
    cudaMemset(&D2_B, 0, sizeof(float) * matB_size);
    cudaMemset(&D2_C, 0, sizeof(float) * matC_size);

    cudaMemset(&D3_A, 0, sizeof(float) * matA_size);
    cudaMemset(&D3_B, 0, sizeof(float) * matB_size);
    cudaMemset(&D3_C, 0, sizeof(float) * matC_size);

    cudaDeviceSynchronize();

    // Operand transition
    cudaMemcpy(D1_A, H1_A, sizeof(float) * matA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(D1_B, H1_B, sizeof(float) * matB_size, cudaMemcpyHostToDevice);

    cudaMemcpy(D2_A, H1_A, sizeof(float) * matA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(D2_B, H1_B, sizeof(float) * matB_size, cudaMemcpyHostToDevice);

    cudaMemcpy(D3_A, H1_A, sizeof(float) * matA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(D3_B, H1_B, sizeof(float) * matB_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Grid, Block\n");
    printf("Grid : (%d, %d), Block : (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // GPU simple matrix multiplication
    timer.onTimer(GPU1);
    simple_matmul<<<gridDim, blockDim>>>(D1_A, D1_B, D1_C, m, n, k);
    cudaDeviceSynchronize();
    timer.offTimer(GPU1);
    cudaMemcpy(G1_C, D1_C, sizeof(float) * matC_size, cudaMemcpyDeviceToHost);

    // GPU shared memory tiling matrix multiplication
    timer.onTimer(GPU2);
    shared_matmul<<<gridDim, blockDim>>>(D2_A, D2_B, D2_C, m, n, k);
    cudaDeviceSynchronize();
    timer.offTimer(GPU2);
    cudaMemcpy(G2_C, D2_C, sizeof(float) * matC_size, cudaMemcpyDeviceToHost);

    // GPU cuBLAS matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f; // scaling factor a
    float beta = 0.0f;  // scaling factor b

    timer.onTimer(GPU3);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, D3_A, m, D3_B, k, &beta, D3_C, m);
    cudaDeviceSynchronize();
    timer.offTimer(GPU3);
    cudaMemcpy(G3_C, D3_C, sizeof(float) * matC_size, cudaMemcpyDeviceToHost);

    // operation check
    bool result1 = true;
    for (int i = 0; i < matC_size; i++)
    {
        if ((H1_C[i] != G1_C[i]) | (H1_C[i] != G2_C[i]))
        {
            printf("[%d] %.2f | %.2f | %.2f \n", i, H1_C[i], G1_C[i], G2_C[i]);
            result1 = false;
        }
    }
    if (result1 == true)
    {
        printf("GPU1, GPU2 matrix multiplication correct\n");
    }
    else
    {
        printf("GPU1 or GPU2 matrix multiplication fault\n");
    }

    // operation check
    bool result2 = true;
    for (int i = 0; i < matC_size; i++)
    {
        if (H2_C[i] != G3_C[i])
        {
            printf("[%d] %.2f | %.2f \n", i, H2_C[i], G3_C[i]);
            result2 = false;
        }
    }
    if (result2 == true)
    {
        printf("GPU3 matrix multiplication correct\n");
    }

    timer.printTimer();
    printf("\n");

    printf("========================Application End========================\n");
    return 0;
}