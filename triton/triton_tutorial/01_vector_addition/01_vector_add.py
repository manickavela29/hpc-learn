import triton
import triton.language as tl

import torch
import time

'''

x_ptr, y_ptr : pointers to the first element
BLOCK_SIZE   : 1) #elements each program should process
               2) 64 then, in thread blocks of [0:64], [65:128], [129: 192]
mask         : mask for gaurding out-of-bounds access

'''

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n,
               BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)

    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) :
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n = output.numel()

    #Grid and block size are calculated similar to CUDA
    # Option 1
    # block_size =  2048
    # grid_size = (n + block_size - 1) // block_size
    # grid = (grid_size,)

    # Option 2
    # meta args <- dynamic for backend to autotune
    grid = lambda meta: (triton.cdiv(n,meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 2**5

x = torch.rand(size,device='cuda')
y = torch.rand(size,device='cuda')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)