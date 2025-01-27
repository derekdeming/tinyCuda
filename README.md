# tinyCuda: CUDA Playground

lets play around and build things using cuda -- name inspired by tinyGrad

We will make this a playground for all things CUDA and while doing that we will do a challenge of 100 days of CUDA.

- **Inspiration**:
  - [tinyGrad](https://github.com/geohot/tinygrad) for the minimalist approach to GPU compute.
  - [hkproj's CUDA collection](https://github.com/hkproj) for daily practice structure.
  - NVIDIA's official CUDA docs and developer resources.

## Contents

1. [Overview](#overview)
2. [Mandatory & Optional Tasks](#mandatory--optional-tasks)
3. [Project Progress by Day](#project-progress-by-day)
4. [How to Load CUDA Kernels into PyTorch](#how-to-load-cuda-kernels-into-pytorch-quick-guide)
5. [How to Load CUDA Kernels into PyTorch (detailed guide)](#how-to-load-cuda-kernels-into-pytorch-detailed-guide)
6. [References & Inspiration](#references--inspiration)

## Overview

- **Goal**: Improve CUDA proficiency through daily hands-on tasks:
  - Writing and optimizing kernels for actual deep learning operations (Softmax, MatMul, Convolution, etc.)
  - Integrating custom CUDA kernels into PyTorch or other frameworks
  - Experimenting with memory layouts, shared memory, tiling, and kernel fusion
  - Building a repository of CUDA examples—from beginner-friendly to advanced training/inference ops

## Mandatory and Optional Tasks

| Day  | Task Description                                                                                                 |
|-----:|:-----------------------------------------------------------------------------------------------------------------|
| D15  | **Mandatory FA2-Forward**: Implement a forward pass for a custom neural network layer (FA2).                    |
| D20  | **Mandatory FA2-Backward**: Implement backward pass for the same neural network layer (gradient computation).    |
| D20  | **Optional Fused Chunked CE Loss + Backward**: Fused kernel for chunked cross-entropy loss with backward pass.   |
| D35  | **Mandatory Softmax Optimization**: Optimize Softmax kernel with shared memory and reduce global sync overhead. |
| D45  | **Optional LayerNorm + Backward**: Fused LayerNorm kernel for training performance.                              |
| D60  | **Mandatory Tiled Convolution**: Implement a 2D convolution with tiling and shared mem.                          |
| D75  | **Optional Flash Attention**: Advanced memory optimization for attention mechanism.                              |
| D90  | **Mandatory Full Transformer Block**: Combine MatMul, Softmax, and feed-forward ops in a single pipeline.        |
| D100 | **Final Project**: End-to-end network inference kernel—integrate everything from Day 1 to Day 99.                |

---

## Project Progress by Day

| Day   | File(s) / Project Name                                                                                                  | Summary & Key Learnings                                                                                                                                                                                                           |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1**  | **helloCUDA.cu** | **Device Setup**: Prints GPU device info and "Hello CUDA!" to confirm your toolchain setup. Learn about CUDA environment setup - Check GPU capabilities (cores, SM count, etc.) |
| **2**  | **vecAdd.cu** | **Vector Addition**: Host-device memory transfers, thread indexing, kernel launches. Memory management (`cudaMalloc`, `cudaMemcpy`) - Thread indexing concepts |
| **3**  | **matrixAdd.cu** | **2D Matrix Addition**: Explores row/column indexing and boundary checks. 2D thread/block mapping - Synchronization basics |
| **4**  | **simpleSoftmax.cu** | **Naive Softmax**: 1D softmax as a stepping stone for advanced attention. Exponent + sum reduction pattern - Potential for shared-memory optimization |
| **5**  | **sharedSoftmax.cu** | **Optimized Softmax**: Shared-memory approach for softmax. Block-wide exponent and reduction - Numerical stability considerations (avoiding overflow) |
| **6**  | **pyExtensionSetup/rollCall.cu** | **Python-CUDA Integration**: Minimal Torch C++ extension example. `torch.utils.cpp_extension.load` usage - Mapping PyTorch tensors to raw pointers |
| **7**  | **naiveMatMul.cu** | **Matrix Multiplication (Naive)**: Global memory reads/writes. Fundamental MatMul loops - Identifying performance bottlenecks |
| **8**  | **tiledMatMul.cu** | **Tiled MatMul**: Shared memory usage for faster matrix multiplication. Block tiling strategy - Shared-memory blocking for coalesced reads |
| **9**  | **layerNorm.cu** | **Layer Normalization**: Parallel mean/variance calculations. Reduction pattern with partial sums - Handling floating-point stability |
| **10** | **elementwiseOps.cu** | **Fused Elementwise Ops**: E.g., ReLU + custom transforms in one kernel. Kernel fusion strategies - Minimizing global memory I/O |
| **11** | **conv1D.cu** | **1D Convolution**: Sliding window with shared memory. Thread-block layout for conv - Handling boundary conditions (halo regions) |
| **12** | **pytorchIntegration/\***.cu**, `.cpp`, `test.py` | **PyTorch Extension**: Launching a custom CUDA kernel from Python. `PYBIND11_MODULE` macros - Dynamic compilation with `setup.py` |
| **13** | **imageOps/color2gray.cu** | **Image Processing**: Convert RGB to grayscale. 2D/3D thread indexing for images - Memory layout considerations |
| **14** | **reduceSum.cu** | **Parallel Reduction**: Summing large arrays in multiple stages. Tree-based reduction, warp divergence considerations - Shared memory usage for partial sums |
| **15** | **FA2Forward.cu** | **Custom Forward Pass (FA2)**: Multi-feature transformations in a single kernel - Lays groundwork for backprop |
| **16** | **atomicAddOps.cu** | **Atomic Operations**: Usage of `atomicAdd` for partial sums/histograms. When to use atomics vs. reductions - Bank conflicts and concurrency |
| **17** | **prefixSum.cu** | **Prefix Sum / Scan**: Classic parallel inclusive/exclusive scan. Block-wise approach, merging partial results - Thread synchronization (`__syncthreads()`) |
| **18** | **fusedBackwardPrep.cu** | **Fused Backward Prep**: Helper kernels for data rearrangement in backprop. Combine multiple reorder ops in a single pass - Memory coalescing in training loops |
| **19** | **softmaxBackward.cu** | **Softmax Backward**: Grad for cross-entropy. Minimizing intermediate buffer usage - Numerical stability with exponent subtraction |
| **20** | **FA2Backward.cu**, **fusedCELoss.cu** (Optional) | **FA2 Backprop / Fused CE Loss**: Implements backward pass for FA2 - Optional chunked cross-entropy kernel referencing Liger approach |
| **21** | **transposeMatrix.cu** | **Matrix Transpose**: 2D data layout transformation. Coalesced accesses for GPU throughput - Handling non-square matrices |
| **22** | **sharedMemTricks.cu** | **Advanced Shared Memory**: Partial sums, sub-tiles, etc. Optimizing block-level shared memory - Bank conflict avoidance |
| **23** | **warpShuffle.cu** | **Warp Shuffle Intrinsics**: In-warp communication. Using `__shfl_xxx` for data exchange among threads in a warp - Reducing global sync overhead |
| **24** | **persistentThreads.cu** | **Persistent Kernel Launches**: Minimizing overhead of frequent launches. Looping in the kernel for repeated tasks - Useful for streaming data or rolling updates |
| **25** | **cublasIntegration.cu** | **cuBLAS Integration**: Compare custom kernels with cuBLAS. High-performance library calls - Evaluate speedups vs. hand-written kernels |
| **26** | **FlashAttentionForward.cu** | **Modern Attention (Forward)**: Memory-efficient patterns. Large-batch matmul + softmax + scaling - Block-based partial sums and caching |
| **27** | **FlashAttentionBackward.cu** | **Modern Attention (Backward)**: Grad of attention. Managing derivative flows for Q/K/V - Ensuring numerical stability |
| **28** | **MultiHeadAttention.cu** | **Multi-Head Attention**: Expand single-head to multi-head. Batch multiple heads in parallel threads - Potential kernel fusion for Q/K/V computations |
| **29** | **BatchedMatMul.cu** | **Batched Matrix Multiply**: Large-scale inference optimization. Launch strategies for B x M x K x N - Performance trade-offs for different tiling techniques |
| **30** | **GPT2StyleOps.cu** | **Language Model Kernels**: GPT-2 style MLP + layernorm fused. Fused feed-forward: GeLU, projection - Minimizing memory usage between layers |
| **31** | **Conv2D_Tiled.cu** | **2D Convolution (Mandatory)**: Tiled approach. Advanced indexing for large feature maps - Shared-memory blocking to reduce global mem accesses |
| **32** | **VisionTransformerForward.cu** | **Vision Transformers**: Patch embedding + self-attention for images. Patchify images in a single kernel - Re-use advanced attention code |
| **33** | **DDPMKernel.cu** (Optional) | **Diffusion Model Step**: GPU kernel for random noise addition. For stable diffusion–like pipelines - Focus on random number generation + partial denoising |
| **34** | **ImageAugmentation.cu** | **Data Augmentations**: Cropping, flipping, color jitter. Coordinate transformations for augmentations - Thread indexing for large image batches |
| **35** | **FusedLayerNormBackward.cu** (Optional) | **LayerNorm Backward (Fused)**: Single-kernel backprop - Block-level partial sums for mean/var |
| **36** | **SparseOps/spMVELL.cu**, **SparseOps/spMVCOO.cu** | **Sparse Matrix Ops**: ELL vs. COO formats. Memory patterns for sparse data - Benchmark vs. dense MatMul |
| **37** | **SparseOps/spMVHybrid.cu** | **Hybrid SpMV**: Combine ELL + COO for coverage. Optimizing data layouts for real-world sparsity - Handling irregular non-zero distributions |
| **38** | **mergeSort.cu** | **Parallel Merge Sort**: Using co-rank approach. Divide & conquer in GPU context - Shared memory for merging sub-blocks |
| **39** | **QuantizationOps.cu** (Optional) | **Quantized Inference**: QAT-friendly kernels. Implement int8 or int4 matmul - Scale/zero-point conversion |
| **40** | **MoEKernel.cu** (Mixture of Experts) | **Gating + Expert Dispatch** (large LLMs). Gating function to route tokens - Parallel dispatch with atomic scatter |
| **...**| *Additional Daily Tasks* | *Expand on HPC, advanced kernel fusion, multi-GPU strategies, domain tasks (audio, speech, 3D), real-time inference.* |
| **75** | **FlashAttention.cu** (Revisited) | **Advanced Tiling**: Full-scale Flash Attention with block-level pipelining.|
| **90** | **FullTransformerBlock.cu** | **Full Forward Pass**: Combine multi-head attn, MLP, layernorm.|
| **100**| **FinalProject.cu** | **End-to-End Inference**: Softmax, LayerNorm, MatMul, etc. Potential small generative model pipeline - Focus on near-production optimization |

---

## How to load into Pytorch (quick guide)

1. Create CUDA kernel file (`.cu`):
   - Define your CUDA kernel with `__global__`
   - Create launch wrapper function to handle grid/block setup
   - Include necessary CUDA/PyTorch headers

2. Create C++ binding file (`.cpp`):
   - Include kernel header file
   - Create tensor wrapper functions
   - Use `PYBIND11_MODULE` to expose functions to Python

3. Load in Python:

    ```python
        from torch.utils.cpp_extension import load
        cuda_module = load(
            name='my_extension',
            sources=['kernel.cu', 'binding.cpp'],
            verbose=True
        )
    ```

4. Use the loaded module:

    ```python
    output = cuda_module.my_kernel(input_tensor)
    ```

---

## How to load CUDA kernels into Pytorch (detailed guide)

While there are many approaches, here's a high-level guide:

1. **Create a Template Kernel** (e.g., `myKernel.cu`):

   ```cpp
   #include <cuda_runtime.h>
   #include <torch/extension.h>
   
   __global__ void myKernelFunction(float* data, int size) {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       if (idx < size) {
           data[idx] = data[idx] + 1.0f; // Example op
       }
   }
   
   void launchMyKernel(torch::Tensor input) {
       const int size = input.numel();
       float* ptr = (float*) input.data_ptr<float>();
       
       // Grid/Block config
       int blockSize = 256;
       int gridSize  = (size + blockSize - 1) / blockSize;
       
       // Launch kernel
       myKernelFunction<<<gridSize, blockSize>>>(ptr, size);
   }
   
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("launch_my_kernel", &launchMyKernel, "My Custom CUDA Kernel");
   }
   ```

2. setup the `.cpp` file (if needed):
    - include the header file
    - implement a c++ func that will call the kernel launch code
    - create a wrapper so that you can use tensors (like `pybind11_module`) for python bindings

3. use setup.py to compile the extension:

    ```python
    from setuptools import setup
    from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

    setup(
        name='my_cuda_extension',
        ext_modules=[
            CUDAExtension('my_cuda_extension', [
                'myKernel.cu', 
                'myKernel.cpp'
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    ```

4. build and load the extension in Python:

    ```python

    import torch
    import my_cuda_extension

    x = torch.ones(1024, device='cuda')
    my_cuda_extension.launch_my_kernel(x)
    print(x)  # Each element should be incremented by 1.0
    ```

---

## References & Inspiration

- [hkproj's CUDA collection](https://github.com/hkproj) for daily practice structure.
- [tinyGrad](https://github.com/geohot/tinygrad) for the minimalist approach to GPU compute.
- [GPU-MODE Lectures](https://github.com/gpu-mode/lectures) for a take on industry and academia professionals on CUDA.
- NVIDIA's official CUDA docs and developer resources.
