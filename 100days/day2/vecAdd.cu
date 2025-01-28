/******************************************************************************
 * Day 2: vecAdd.cu
 *
 * we want to:
 * 1. Allocate and transfer data between host and device
 * 2. An elementwise vector addition on the GPU
 * 3. Basic err checks for CUDA calls
 * 
 * for mem allocation we will use `cudaMalloc` and `cudaFree`
 * for copying data we will use `cudaMemcpy`
 * for kernel launches we will use `<<<>>>`
 * for synchronization we will use `cudaDeviceSynchronize`
 ******************************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
// #include <cstdlib>

#define CHECK_CUDA_ERR(err) checkCudaError((err), #err, __FILE__, __LINE__)

void checkCudaError(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << func << " in " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * kernel for vec addition
 * each thread will add one element from vector a and one from vector b and will 
 * write to c
 */
__global__ void vecAddKernel(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * we need to verify the result on the host (cpu) - basically the cpu baseline 
 */
void vecAddBaseline(float* a, float* b, float* c, int n) {
    for (size_t i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (abs(c[i] - expected) > 1e-5) {
            std::cerr << "Error at index " 
                      << i << ": expected " 
                      << expected << ", got " 
                      << c[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Verification passed. Results match cpu baseline" << std::endl;
}

int main() {
    
    int n = 1 << 20; // this is vector size 2^20 = 1048576
    
    // host stuff 
    std::vector<float> h_A(n), h_B(n), h_C(n); // host vectors

    // initialize host vectors
    for(int i = 0; i < n; ++i) {
        // random float between 0 and 1. the static cast is to convert the int to float
        h_A[i] = static_cast<float>(rand()) / RAND_MAX; 
        h_B[i] = static_cast<float>(rand()) / RAND_MAX; 
    }

    // device stuff

    float *d_A, *d_B, *d_C; // device pointers for vectors

    CHECK_CUDA_ERR(cudaMalloc(&d_A, n * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_B, n * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_C, n * sizeof(float)));

    // copy data from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_A, h_A.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_B, h_B.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel params
    int blockSize = 256;  // number of threads per block
    
    // Calculate number of blocks needed:
    // - If n divides evenly by blockSize, we get the exact number of blocks needed
    // - If n doesn't divide evenly, we need to round up to ensure we have enough threads
    // Example: n=1000, blockSize=256
    // - 1000 + 255 = 1255
    // - 1255 / 256 = 4.9 -> rounds up to 5 blocks 
    // - This gives us 5 blocks * 256 threads = 1280 total threads, enough to cover n=1000
    int gridSize = (n + blockSize - 1) / blockSize;  // ceil(n/blockSize)

    // launch kernel
    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // check for any errors in the launch 
    CHECK_CUDA_ERR(cudaGetLastError());

    // synchronize
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // copy result back to host
    CHECK_CUDA_ERR(cudaMemcpy(h_C.data(), d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

    // verify result
    vecAddBaseline(h_A.data(), h_B.data(), h_C.data(), n);

    // free device memory
    CHECK_CUDA_ERR(cudaFree(d_A));
    CHECK_CUDA_ERR(cudaFree(d_B));
    CHECK_CUDA_ERR(cudaFree(d_C));

    std::cout << "GPU vecAdd completed successfully!\n";
    return 0;
}
