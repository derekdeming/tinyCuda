/******************************************************************************
 * Day 1: helloCUDA.cu
 *
 * Demonstrates:
 * 1. Querying CUDA-capable device info.
 * 2. A kernel launching and printing "Hello CUDA!" from the device.
 ******************************************************************************/

#include <cuda_runtime.h>
#include <iostream>

/**
 * kernel that prints "Hello CUDA!" from each thread. We will be adding a ton of print 
 * statements to see how the kernel is executed. This will add overhead to the kernel execution.
 * but this is for learning purposes and to see how the kernel is executed.
 */

__global__ void helloKernel()
{
    // first calc the unique thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello CUDA! From block %d, thread %d, global thread ID %d\n", blockIdx.x, threadIdx.x, idx);
}

void printDeviceProperties()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "Num of CUDA capable devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout << "device " << dev << " name: " << deviceProp.name << "\n";
        std::cout << "  total global memory:     "
                  << (deviceProp.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  shared memory per block:   "
                  << deviceProp.sharedMemPerBlock << " bytes\n";
        std::cout << "  registers per block:       "
                  << deviceProp.regsPerBlock << "\n";
        std::cout << "  warp size:                 "
                  << deviceProp.warpSize << "\n";
        std::cout << "  max threads per block:   "
                  << deviceProp.maxThreadsPerBlock << "\n";
        std::cout << "  multiprocessor count:   "
                  << deviceProp.multiProcessorCount << "\n";
        std::cout << "  compute possible:      "
                  << deviceProp.major << "." << deviceProp.minor << "\n\n";
    }
}

/**
 * main func
 * - this prints device info and then launches the kernel
 */

int main()
{
    //print info about the device
    printDeviceProperties();

    // config a small grid of threads to demo the kernel print
    dim3 blockSize(4, 1, 1); // 4 threads in a block
    dim3 gridSize(2, 1, 1); // 1 block in a grid

    helloKernel<<<gridSize, blockSize>>>();

    // wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch or execution w/ error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Kernel execution completed." << std::endl;
    return 0;
}