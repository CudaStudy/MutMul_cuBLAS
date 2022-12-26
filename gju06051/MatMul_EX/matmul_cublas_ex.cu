// This program is cloning below youtube lecture about
// calculating matrix multiplication using SGEMM cuBlas
// by Nick from CoffeBeforeArch
// https://www.youtube.com/watch?v=MVutNZaNTkM

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

void verify_solution(float *_matA, float *_matB, float *_matC, int _n)
{
    float temp;
    float epsilon = 0.001;

    for (int i = 0; i < _n; i++)
    {
        for (int j = 0; j < _n; j++)
        {
            temp = 0;
            for (int k = 0; k < _n; k++)
            {
                temp += _matA[k * _n + i] * _matB[j * _n + k];
            }
            assert(fabs(_matC[j * _n + i] - temp) < epsilon);
        }
    }
}

int main(void)
{
    // problem size
    int n = 1 << 10;
    size_t bytes = n * n * sizeof(float);

    // declare pointer of matrix
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // allocate memory
    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Pseudo random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the rand seed
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Fill the matrix with random numbers on the device
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scaling factors -> just for matrix multiplication
    float alpha = 1.0f;
    float beta = 0.0f;

    // Calculate: c = (alpha * a) * b + (beta * c)
    // (m x n) * (n * k) = (m x k)
    // Signature : handle, operation, operatin, m, n, k, alpha, A, lda, B, ldb, beta, C, ldC)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    // Operation
    // CUBLAS_OP_N : normal
    // CUBLAS_OP_T : tranpose

    // ldX : leading dimension of X

    // Copy back the three matrix
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify solution
    verify_solution(h_a, h_b, h_c, n);
    printf("Completed successfully\n");

    return 0;
}