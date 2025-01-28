# Day 1: Hello CUDA!

i first want to make sure we have everything necessary to execute CUDA code so
this simple "hello cuda" example will suffice:

1. Query and display info about my "CUDA-capable" device  
2. we launch our CUDA kernel that prints "Hello CUDA!" from each thread
3. i figured we can print out some info about the device + threads

> **Disclaimer:**
> i'm using a windows machine and i will use `o1` / `o1pro` to help write the README docs related to the `.cu` files for each day
>
> ideally this serves as a combination of theory + implementation but we shall see
>
> i'm not sure if this is the best way to do this but i'm just trying to provide as much context as possible while not completely wasting my time writing documentation
>
> i will try to keep the README docs as short as possible but i will also try to provide as much context as possible

---

> **Note:**
> i'm using the `o1` / `o1pro` to help write the below section by giving it my `.cu` file and asking it to generate the README docs

## 1. Device Query

The code uses `cudaGetDeviceCount` and `cudaGetDeviceProperties` to gather information about the system's available GPU devices. For each device, it prints key hardware specifications like:  
- Global memory  
- Shared memory per block  
- Registers per block  
- Warp size  
- Maximum threads per block  
- Number of multiprocessors  
- Compute capability (major.minor)

These details help you understand the GPU resources available for parallel execution.

---

## 2. Kernel Execution

The CUDA kernel (`helloKernel`) is launched using the `<<<gridSize, blockSize>>>` syntax. Each thread prints a line that identifies its block and thread coordinates, along with a unique global thread ID (calculated as `blockIdx.x * blockDim.x + threadIdx.x`).

### Thread Hierarchy

- The grid consists of multiple blocks (`gridSize`).
- Each block contains multiple threads (`blockSize`).
- In this example:  
  - `gridSize = (2, 1, 1)` → 2 blocks.  
  - `blockSize = (4, 1, 1)` → 4 threads per block.  

Hence, 8 total threads will run the `helloKernel`, each printing a message with its ID.

---

## 3. GPU Synchronization

After the kernel is launched, the host calls `cudaDeviceSynchronize()` to ensure all threads on the GPU have finished before proceeding. If any errors occur during execution, they are captured and logged.

---

## 4. Summary

- Printing device properties shows you the GPU's capabilities and constraints.  
- Launching a simple kernel with minimal computations (printing output) introduces the concept of thread indexing in CUDA.  
- Synchronization ensures that your program waits until GPU tasks are complete.

This foundational example provides a glimpse of how CUDA code is structured—starting from device queries and extending to kernel configuration and execution.
