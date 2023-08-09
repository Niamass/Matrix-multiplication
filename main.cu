#include "multMatrix.cuh"
#include <random>

int main()
{
    const size_t m = 500, n = 500, l = 500;
    float* A, * B, * C; // A: m x n, B: n x l

    A = new float[m * n]();
    B = new float[n * l]();
    C = new float[m * l]();

    for (size_t i = 0; i < m * n; ++i)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < n * l; ++i)
    {
        B[i] = (float)rand() / RAND_MAX;
    }

    cudaError_t cudaStatus = multMatrixCuda(C, A, B, m, n, l, SIMPLE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrix multiplication failed!");
        return 1;
    }

    //printMatrix(A, m, n);
    //printMatrix(B, n, l);
    //printMatrix(C, m, l);
    std::cout << isCorrect(C, A, B, m, n, l) << std::endl;

    delete[]A;
    delete[]B;
    delete[]C;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}