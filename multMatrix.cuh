#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>

enum multMatrixMode {
    SIMPLE,
    SHARED_MEMORY,
    WARP_INTRINSICS,
};

cudaError_t multMatrixCuda(float* C, const float* A, const float* B, const size_t m, const size_t n, const size_t l, multMatrixMode mode);

float isCorrect(float* C, float* A, float* B, const size_t m, const size_t n, const size_t l);
void multMatrix(float* C, const float* A, const float* B, const size_t m, const size_t n, const size_t l);
void printMatrix(float* matrix, size_t rows, size_t cols);