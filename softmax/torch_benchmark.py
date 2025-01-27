import torch
import time

# testing different matrix sizes
matrix_sizes = [
    (128, 1024),
    (256, 2048), 
    (512, 4096),
    (1024, 8192),
    (1024, 16384),
    (1024, 32768), 
    (1024, 65536),
    (1024, 131072),
    (1024, 262144),
    (1024, 524288),
    # (1024, 1048576), # 1024 x 1048576 = 1073741824 elements this fails due to outofmemory error trying to allocate 4GB of memory
]

n_iter = 5 # number of iterations to average the time over
for rows, cols in matrix_sizes:
    print(f"\nTesting matrix size: {rows} x {cols}")
    
    # create matrix on cuda
    print(f"Creating matrix on cuda with size {rows} x {cols}")
    # ensuring we're measuring GPU performance and not including any overhead from data transfers
    matrix = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
    
    ''' warm up -- "priming" the GPU so that the first measured operation does not include 
    the overhead of the first run (e.g. GPU startup time or kernel initialization time) '''
    _ = torch.nn.functional.softmax(matrix, dim=-1)
    
    # measure time
    total_time = 0
    for i in range(n_iter):
        ''' synchronize is used before and after the operation to ensure that the operation is 
        finished before measuring the time '''
        torch.cuda.synchronize()  # check if all CUDA operations are finished
        start = time.time()
        _ = torch.nn.functional.softmax(matrix, dim=-1)
        torch.cuda.synchronize()  # synchronize again
        end = time.time()
        
        total_time += (end - start) * 1000
        
    avg_time = total_time / n_iter
    print(f"Matrix size {rows}x{cols} - Average time: {avg_time:.2f} ms")
    print(f"Elements per second: {(rows * cols) / (avg_time / 1000):,.0f}")


'''
Results: 
1. for very small matrices (eg 128x1024, 256x2048), the avg. time is extremely short. 
   However, the throughput can look decptively high bc there is less data to process and some constant 
   overhead in launching GPU kernels 
   
2. the highest throughput (~21e9 elements/s) is achieved for a matrix size of 1024x8192 and 1024x16384. This is 
   likely due to the fact that the GPU can efficiently handle these sizes and the overhead of launching and is the 
   "sweet spot" for GPU performance.
   - enough data is being process to keep the GPU cores busy 
   - the overhead of launching and synchronizing GPU kernels is minimized compared to the computation time

3. throughput drops for very large matrices (eg 1024x32768 and above) from 21e9 elements/s to 15e9 elements/s. 
   This is likely due to: 
   - increased mem bandwidth pressure
   - larger datasets not fitting nicely into caches causing memory stalls
   - more gpu launch and synchronization overhead

4. something to note is that the torch.cuda.synchronize() call is necessary for benchmarking but adds a small overhead
   to the operation. This would impact small matrices proportionally more, but for very large matrices, the overhead 
   is negligible because memory bandwidth and kernel execution time dominates

** to note that this is on a T4 GPU, so the results may be different on a different GPU ** 

'''