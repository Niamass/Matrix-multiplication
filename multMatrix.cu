#include "device_launch_parameters.h"
#include "multMatrix.cuh"

constexpr auto BLOCK_SIZE = 32;

__global__ void multMatrixKernel(float* C, const float* A, const float* B, const size_t m, const size_t n, const size_t l)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (idx_y >= m || idx_x >= l)
    {
        return;
    }

    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j)
    {
        sum += A[idx_y * n + j] * B[j * l + idx_x];
    }
    C[idx_y * l + idx_x] = sum;
}


__global__ void multMatrixSharedMemKernel(float* C, const float* A, const float* B,const size_t m, const size_t n, const size_t l)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y; 
    int idx_x = blockDim.x * blockIdx.x + tx;
    int idx_y = blockDim.y * blockIdx.y + ty;
    
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += BLOCK_SIZE)
    {
        shared_A[ty][tx] = (idx_y * n + i + tx < m * n) ? A[idx_y * n + i + tx] : 0.0f;
        shared_B[ty][tx] = (idx_x + (i + ty) * l < n * l) ? B[idx_x + (i + ty) * l] : 0.0f;

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        __syncthreads();
    }

    if (idx_y >= m || idx_x >= l)
    {
        return;
    }

    C[idx_y * l + idx_x] = sum;
}


__global__ void multMatrixWarpIntrKernel(float* C, const float* A, const float* B, const size_t m, const size_t n, const size_t l)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + tx;
    int idx_y = blockDim.y * blockIdx.y + ty;

    float sum = 0.0f;
    float val_A;
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    for (size_t i = 0; i < n; i += warpSize)
    {
        val_A = (idx_y * n + i + tx < m * n) ? A[idx_y * n + i + tx] : 0.0f;
        shared_B[ty][tx] = (idx_x + (i + ty) * l < n * l) ? B[idx_x + (i + ty) * l] : 0.0f;
        //val_A = A[idx_y * n + i + tx];
        //val_B = B[idx_x + (i + ty) * l];
        __syncthreads();

        for (size_t k = 0; k < warpSize; ++k)
        {
            //val_B = (idx_x + (i + k) * l < n * l) ? B[idx_x + (i + k) * l] : 0.0f;
                //sum += __shfl_sync(0xffffffff, val_A, k) * val_B;
            //sum += __shfl_sync(0xffffffff, val_A, k) * B[idx_x + (i + k) * l];
            sum += __shfl_sync(0xffffffff, val_A, k) * shared_B[k][tx];
        }
        __syncthreads();
    }

    if (idx_y >= m || idx_x >= l)
    {
        return;
    }

    C[idx_y * l + idx_x] = sum;
}


cudaError_t multMatrixCuda(float* C, const float* A, const float* B, const size_t m, const size_t n, const size_t l, multMatrixMode mode)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_c, m * l * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, m * n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, n * l * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, B, n * l * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(ceil((float)l / block_dim.x), ceil((float)m / block_dim.y));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    switch (mode)
    {
    case SIMPLE:
        multMatrixKernel<<< grid_dim, block_dim >>>(dev_c, dev_a, dev_b, m, n, l);
        break;
    case SHARED_MEMORY:
        multMatrixSharedMemKernel <<< grid_dim, block_dim >> > (dev_c, dev_a, dev_b, m, n, l);
        break;
    case WARP_INTRINSICS:
        multMatrixWarpIntrKernel <<< grid_dim, block_dim >> > (dev_c, dev_a, dev_b, m, n, l);
        break;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time: %.2f millseconds\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, dev_c, m * l * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}



void printMatrix(float* matrix, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows * cols; ++i) {

        std::cout << matrix[i] << " ";
        if (i % cols == cols - 1)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

void multMatrix(float* C, const float* A, const float* B, const size_t m, const size_t n, const size_t l)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t k = 0; k < l; ++k)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < n; ++j)
            {
                sum += A[i * n + j] * B[j * l + k];
            }
            C[i * l + k] = sum;
        }
    }
}

float isCorrect(float* C, float* A, float* B, const size_t m, const size_t n, const size_t l)
{
    float* answer = new float[m * l]();
    multMatrix(answer, A, B, m, n, l);
    //printMatrix(answer, m, l);

    float eps, max_eps = 0;
    for (size_t i = 0; i < m * l; ++i)
    {
        eps = fabs(C[i] - answer[i]) / answer[i];
        if (eps > max_eps)
            max_eps = eps;
    }
    return max_eps;
}

