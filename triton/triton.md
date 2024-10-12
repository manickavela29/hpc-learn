# Triton

Triton is DSL(Domain Specific Language) designed for GPU programming.
**MLIR**  felxible compiler infrastructure plays a crucial role to performaoptimization  at different stages,Higher level tensor operations are lowered progressively into lower-level optimized representation at each level

Triton goes ahead SPMD execution model, paradigm based on blocked algorithms.

CUDA   : Scalar program, Blocked threads
Triton : Blocked Program, Scalar threads

### PROs
- High level programming language compared to CUDA , extream ease of use
- User only defines the computation operation over tensors, it will autmatically manage thread scheduling(thread, blocks, warps) and memory managements
- Reaches on par performance with CUDA but most of the times lesser

### CONs
- Performance is not tunable, due to lack of control.
  - Minimal support by hints for tile sizes and scheduling