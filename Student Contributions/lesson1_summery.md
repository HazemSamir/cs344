#LESSON 1 - The GPU Programming Model

### *Quiz* What are the 3 traditional ways HW designers make computers run faster?
1. faster clocks.
2. more clock/clock cycle.
3. more processors.

### Modern GPU:
- Thousands of ALUs.
- Hundreds of processors.
- Tens of thousands of concurrent Threads.

### CPU vs GPU:

#### CPU:
- complex control hardware.
- high flexibility + performance.
- expensive in terms of power.
- optimizes for *Latency* (time to finish task)(unit like second).

#### GPU:
- simpler control hardware.
- more HW for computation.
- potentially more power efficient.
- more restrictive performance model (explicitly parallel programming model).
- optimizes for *Throughput* (tasks per time)(unit like jobs/hours, pixels/second).

### CUDA Program:
- The device model used is called Heterogeneous.
- CPU (host) is in charge of GPU (device). It runs the main program and sends directions to GPU to tell it what to do:
    - moving data from CPU's memory to GPU's memory (cudaMemCpy).
    - moving data from GPU's memory to CPU's memory (cudaMemCpy).
    - Allocate GPU memory (cudaMalloc).
    - launch kernel (programs that do stuff in parallel) on GPU.
- CUDA assumes CPU and GPU has their separate memory.
- GPU cannot initiate kernels nor requests to the CPU.

### Typical GPU Program:
1. CPU allocates storage on GPU [cudaMalloc].
2. CPU copies input data from CPU to GPU [cudaMemCpy].
3. CPU launches kernel(s) on GPU to process the data.
4. CPU copies resualts back to CPU from GPU.


### GPU good at:
- Launch huge number of threads.
- run those threads in parallel.

### First Cuda program:
- use this convention in naming variables to avoid errors:
    - `h_` prefix to name variable in the memory attached to the CPU (host).
    - `d_` prefix to name variable in the memory attached to the GPU (device).
- CUDA C/C++ keyword `__global__` indicates a function that:
    - Runs on the device
    - Is called from host code
- nvcc separates source code into host and device components
    - Device functions (e.g. mykernel()) processed by NVIDIA compiler
    - Host functions (e.g. main()) processed by standard host compiler
- Triple angle brackets `<<< >>>` mark a call from host code to device code, also called a “kernel launch”.
- code snippet:
    ``` cpp
    __global__ void foo(parameters){
    	// kernel function
    }

    int main(int argc, char ** argv) {

    	// allocate GPU memory
    	cudaMalloc((void**) &d_arr, ARRAY_BYTES);

    	// transfer the array to the GPU
    	cudaMemcpy(d_arr, h_arr, ARRAY_BYTES, cudaMemcpyHostToDevice);

    	// launch the kernel
    	foo<<<number_of_blocks, number_of_threads_per_block>>>(parameters);

    	// copy back the result array to the CPU
    	cudaMemcpy(h_arr, d_arr, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    	cudaFree(d_arr);
    }
    ```
#### Configuring kernel launch:
```
kernel<<< grid of blocks, block of threads, shared memory>>> (...)
```
- both grid of blocks and blocks of threads can be declared to be up to 3 dimensions.
- default is 1 dimension and if not declared default value is 1.
- default shared memory is zero.
- to decalre 3D grid of blocks or threads we use c struct called dim3 **general form:**
```
kernel<<<dim3(bx,by,bz), dim3(tx,ty,tz), shmem>>> (...)
```
- for each thread:
    - thread index in block through struct `threadIdx.x, threadIdx.y, threadIdx.z`.
    - block dimension (size of block) through struct `blockDim.x, blockDim.y, blockDim.z`.
    - block index in grid through struct `blockIdx.x, blockIdx.y, blockIdx.z`.
    - grid dimension (size of grid) through struct `gridDim.x, gridDim.y, gridDim.z`.
- max number of threads per block in old GPUs is 512, in modern ones is 1024.
