import torch

import triton
import triton.language as tl
from triton.runtime import driver


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

"""
Naive implementation of softmax with pytorch

Row-wise softmax on X of shape [M,N]
"""
def naive_softmax(x):
    # x_max dim -> [M], M elements coming from each rows(vectors) max value
    # [0] : torch max returns values and thier indeces, we are only picking the values 
    x_max = torch.max(x, dim=1)[0]      # MN reads , M writes
    
    #[:,None] <- adds additional dimenion to the x_max, so that sub happens
    z = x - x_max[:,None]               # MN + N reads, MN writes 
    
    # numerator dim [M,N] 
    numerator = torch.exp(z)            # MN reads, MN writes
    
    # denominator dim [M]
    denominator = numerator.sum(dim=1)  # MN reads, M writes
    
    ret = numerator/denominator[:,None] # MN + N reads, MN writes

    return ret
    # DRAM and memory access calculation :
    # Total reads  : 5MN + 2M
    # Total writes : 3MN + 2M

"""
Triton softmax kernel
1. Each program loads a set of rows of the input matrix X strided by number of programs(thread_groups),
    normalizes it and writes them back.
2. 
"""

@triton.jit()
def softmax_kernel(output_ptr, input_ptr,
                   input_row_stride, output_row_stride,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr) :
    row_start = tl.program_id(0)
    row_step  = tl.num_programs(0)
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages) :
        # stride - value to increase the pointer to next row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # BLOCK_SIZE : next power of two greater than n_cols,
        #              so that each row fit in block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        row_input_ptr = row_start_ptr + col_offsets

        # mask : handles BLOCK_SIZE > n_cols
        mask = col_offsets < n_cols

        # loads the row specified by row_input_ptr (under row_start_ptr which is spanning several rows) to  SRAM
        row = tl.load(row_input_ptr, mask=mask, other=-float('inf'))

        row_minux_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minux_max)
        denominator = tl.sum(numerator, axis=0)

        softmax = numerator / denominator

        # Write back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        row_output_ptr = output_row_start_ptr + col_offsets

        tl.store(row_output_ptr, softmax, mask=mask)

"""
Below helper functions enqueues the kernel and its meta-arguement for input tensors
"""

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x) :
    n_rows, n_cols = x.shape
    
    # BLOCK_SIZE : for each loop iteration, block size should conatain the row length within
    #              therefore fixing it as a power of size greater than given number
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Increases number of threads used by warp over which row is distributed
    num_warps = 8
    
    num_stages = 4 if BLOCK_SIZE > 200000 else 2
    y = torch.empty_like(x)
    
    # pre-compule kernel to get register usage and compute thread occupancy
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1,))
        
        kernel._init_handles()
        n_regs  = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2
                
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE]  = (kernel, num_programs)
        
        print(f" occupancy : {occupancy}, num_programs : {num_programs}")
    num_programs = min(num_programs, n_rows)
    
    # Creates number of persistent programs
    kernel[(num_programs, 1, 1)](
        y, x,
        x.stride(0),
        y.stride(0),
        n_rows, n_cols,
    )
    
    return y

# Unit test for Softmax implementation

torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)

assert torch.allclose(y_triton, y_torch), (y_triton,y_torch)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)