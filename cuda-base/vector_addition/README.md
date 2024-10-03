## Simple Vector Addition

CPU | GPU  
--- | --- 
45.92 ms | 0.24 ms

## Vector addition with blocked Kernels

Vector addition with grid and block demonstrates the advantage of memory coalescing

Gird to Data Mapping wiht 2D and 3D
-----------------------------------

Grid size is predominantly decided by the dataset and it is mapped based on the thread block dim that is decided optimilly.

Number of blocks is calculated with dataset dimension (nx,ny and nz), each blockk hanldes a subregion of the vector defined by nx,ny,nz

Efficient Distribution of the work load is handled per problmm by mazimizing warm efficienc, memory coalescing and occupancy with the type of dataset


Performance with GRID
---------------------

1. Memory stored in row-major format will be read from nx, ny and nz indexing, with nx moving faster linearly across x-dim, ny moving only for offset y-dim fast hanlding offset for x-dim and nz moving the slowest saong zdim.

2. Thread Performance and warp utiilzation
   1. Larger nx - Increase in #elements along x-dim results in more blocks in x-dim. More blocks more potential for parallel execution
   2. Smaller nz,  z-dim will not fully utilize the 3D grid and result in under utilization
   3. But, when nz increases more threads get distributed along z-dim. z-dim typically corresponds to fewer contiguous memory access leading to poor Warp utilization as memory access pattern becomes scattered. whereas along x and y has more coalesced memory
   4. Data must be well paritioned such that threads in warp accesss consecutive memory locations so that coalesced access happens for better performance
   5. Hardware utilization - z-axis is often capped with lower thread count and it will reduce overal utilization and occupancy once it crosses the threshold
   6. Kernel launch overhead - z-dim large means more launching of kernels, increase in workload and scheduling for managing blocks.

Best GRd Practice 
-----------------
1. Favor the x-dim, better warp utilization and memory coalescing.
2. Balance dimension - if data is flexible then prefer x and y dim , wihh less for z-dim
3. Grid and Block tuning - Tune block_size_3d to ensure maximum occupancy expecially along x-dim wjere threads are well utilized. (32,1,1) or (16,16,1) is lot better than increasing z-axis



<table>
<tr><th> 10 Million data</th><th> 100 Million data</th></tr>
<tr><td>

Grid Dim | CPU | GPU 1D | GPU 3D 
---------|-----|--------|------
(16,8,8) | 45.1 ms | 0.32 ms | 0.33 ms 

</td><td>

Grid Dim | CPU | GPU 1D | GPU 3D 
---------|-----|--------|------
(16,8,8) | 45.1 ms | 0.32 ms | 0.33 ms 

</td></tr> </table>

Experiments and Observations
---------------------------
1. Thread Block Dimension
   1. Order of thread block dims doesn't matter much with performance
      1. i.e (16,8,8) ~ (8,8,16)
   2. Diffreent dimenion also didn't matter much with performance 
      1. i.e (16,8,8) ~ (32,8,4) ~ (64,4,4)