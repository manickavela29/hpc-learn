Triton Performance features


### Flexibility and Dynamism for Kernel

Vector add example with BLOCK_SIZE

2 major things that are happening here
1. Deferred grid calculation : Grid is not calculated until the kernel is launched. Only at launch time ,Triton populatest meta dictionary config with optimal value for BLOCK_SIZE
2. Dynamic block size:: Triton choose best block size based on GPU architecture or input data size. ;ambda function will then calculate the grid size based on the optimal value.


No of elements : 134217728.0
(between pytorhc and triton smaller sizes were almost same, only increaseing showed the difference)

block_size     | triton   | torch
-------------  | ------   | ------
64             | 41.5 ms  | 487.25     
128             | 404.49 ms  | 487.4     
256             | 444.8 ms  | 487.25     
512             | 468.6 ms  | 487.25     
1024            | 479.6 ms  | 487.25     
2048            | 959.06 ms  | 487.4 

*block_size 32 or lesser threw illigal memory access error

Observations: 
1. Lower block size seems to be increasing occupancy and increasing performance
2. increasing chance for memory coalescing, reduce register requirement pressure and easy memory optimizaiton