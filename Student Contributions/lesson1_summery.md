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
- CPU (host) is in charge of GPU (device) sending direction to GPU to tell it what to do:
    - moving data from CPU's memory to GPU's memory (cudaMemCpy).
    - moving data from GPU's memory to CPU's memory (cudaMemCpy).
    - Allocate GPU memory (cudaMalloc).
    - launch kernel on GPU.
- CUDA assumes CPU and GPU has their separate memory.
