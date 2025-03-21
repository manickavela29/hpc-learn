import torch

import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target() ==  "cuda"


triton.jit
def matmul_kernel(
    # pointers to matrix
    a_ptr, b_ptr, c_ptr,
    # Matrix dimesntions
    M, N, K
    # Stride across different dimensions of matrix
    # Represents how much to increase the ptr to move by 1 element in particular dim
    # Example stride_am, value to increaset a_ptr to access next element in M dimension
    
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K, tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Map program id's to the block of C it should compute
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m  = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # block level pointers for the first blocks of A and B
    # these pointers will be advanced as we move in k direction and accumulate
    
    # a_ptrs : pointer for [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs : pointer for [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    
    offs_am  = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn  = (pid_n * BLOCK_SIZe_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    