import ctypes
import numpy as np

# Load NVIDIA driver and create CUDA context
nvcuda = ctypes.WinDLL("nvcuda.dll")
nvcuda.cuInit(0)
device = ctypes.c_int()
nvcuda.cuDeviceGet(ctypes.byref(device), 0)
context = ctypes.c_void_p()
nvcuda.cuCtxCreate(ctypes.byref(context), 0, device)

# Set matrix dimension and allocation size
N = 16  # Matrix is NxN
size = N * N * ctypes.sizeof(ctypes.c_float)

# Allocate GPU memory for matrices A, B, and C
gpu_A = ctypes.c_void_p()
gpu_B = ctypes.c_void_p()
gpu_C = ctypes.c_void_p()
nvcuda.cuMemAlloc(ctypes.byref(gpu_A), size)
nvcuda.cuMemAlloc(ctypes.byref(gpu_B), size)
nvcuda.cuMemAlloc(ctypes.byref(gpu_C), size)

# Create host matrices A and B with specific initial values
A = np.ones((N, N), dtype=np.float32)
B = np.ones((N, N), dtype=np.float32)

# Copy host arrays A and B into GPU memory
nvcuda.cuMemcpyHtoD(gpu_A, A.ctypes.data_as(ctypes.c_void_p), size)
nvcuda.cuMemcpyHtoD(gpu_B, B.ctypes.data_as(ctypes.c_void_p), size)

# Define execution configuration for a 16x16 matrix
block_size = (16, 16, 1)
# Calculate grid size to cover entire matrix
grid_size = ((N + block_size[0] - 1) // block_size[0],
             (N + block_size[1] - 1) // block_size[1],
             1)

# Initialize matrices with meaningful values
A = np.ones((N, N), dtype=np.float32)  # All ones
B = np.ones((N, N), dtype=np.float32)  # All ones
for i in range(N):
    for j in range(N):
        A[i, j] = 1.0  # or any other initialization
        B[i, j] = 1.0  # or any other initialization

# Modified kernel with boundary checks
tensor_kernel = b"""
.version 7.0
.target sm_80
.address_size 64

.visible .entry matmul(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_N
)
{
    .reg .u64    %rd_A, %rd_B, %rd_C;
    .reg .u32    %rN;
    .reg .u32    %row, %col, %k, %temp;
    .reg .f32    %f_sum, %f_A, %f_B;
    .reg .pred   %p_valid;

    ld.param.u64    %rd_A, [param_A];
    ld.param.u64    %rd_B, [param_B];
    ld.param.u64    %rd_C, [param_C];
    ld.param.u32    %rN,   [param_N];

    mov.u32 %row, %ctaid.x;
    mul.lo.u32 %row, %row, %ntid.x;
    add.u32 %row, %row, %tid.x;

    mov.u32 %col, %ctaid.y;
    mul.lo.u32 %col, %col, %ntid.y;
    add.u32 %col, %col, %tid.y;

    setp.ge.u32 %p_valid, %row, %rN;
    @%p_valid bra END;
    setp.ge.u32 %p_valid, %col, %rN;
    @%p_valid bra END;

    mov.f32 %f_sum, 0f00000000;

    mov.u32 %k, 0;
LOOP:
    setp.ge.u32    %p_valid, %k, %rN;
    @%p_valid bra    LOOP_END;

    mul.lo.u32    %temp, %row, %rN;
    add.u32       %temp, %temp, %k;
    mul.wide.u32  %rd_A_offset, %temp, 4;
    add.u64       %rd_A_elem, %rd_A, %rd_A_offset;
    ld.global.f32 %f_A, [%rd_A_elem];

    mul.lo.u32    %temp, %k, %rN;
    add.u32       %temp, %temp, %col;
    mul.wide.u32  %rd_B_offset, %temp, 4;
    add.u64       %rd_B_elem, %rd_B, %rd_B_offset;
    ld.global.f32 %f_B, [%rd_B_elem];

    fma.rn.f32    %f_sum, %f_A, %f_B, %f_sum;

    add.u32 %k, %k, 1;
    bra LOOP;
LOOP_END:

    mul.lo.u32    %temp, %row, %rN;
    add.u32       %temp, %temp, %col;
    mul.wide.u32  %rd_C_offset, %temp, 4;
    add.u64       %rd_C_elem, %rd_C, %rd_C_offset;
    st.global.f32 [%rd_C_elem], %f_sum;
END:
    ret;
}
"""

# Load module and get kernel function ("matmul" entry point)
module = ctypes.c_void_p()
kernel_func = ctypes.c_void_p()
nvcuda.cuModuleLoadData(ctypes.byref(module), tensor_kernel)
nvcuda.cuModuleGetFunction(ctypes.byref(kernel_func), module, b"matmul")

# Prepare kernel parameters by creating local variables holding GPU addresses
param_A = ctypes.c_void_p(gpu_A.value)
param_B = ctypes.c_void_p(gpu_B.value)
param_C = ctypes.c_void_p(gpu_C.value)
n_param = ctypes.c_uint(N)

kernel_params = (ctypes.c_void_p * 4)(
    ctypes.cast(ctypes.byref(param_A), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(param_B), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(param_C), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(n_param), ctypes.c_void_p)
)

# Perform multiple matrix multiplications
num_iterations = 5
for i in range(num_iterations):
    # Launch the kernel
    nvcuda.cuLaunchKernel(kernel_func,
                          grid_size[0], grid_size[1], grid_size[2],
                          block_size[0], block_size[1], block_size[2],
                          0, None, kernel_params, None)
    nvcuda.cuCtxSynchronize()

    # Copy the result matrix C from GPU to host
    C = np.empty((N, N), dtype=np.float32)
    nvcuda.cuMemcpyDtoH(C.ctypes.data_as(ctypes.c_void_p), gpu_C, size)
    print(f"Result matrix C after iteration {i+1}:")
    print(C)

    # Copy result back to A for the next iteration
    nvcuda.cuMemcpyDtoD(gpu_A, gpu_C, size)

# Free GPU memory and destroy the context
nvcuda.cuMemFree(gpu_A)
nvcuda.cuMemFree(gpu_B)
nvcuda.cuMemFree(gpu_C)
nvcuda.cuCtxDestroy(context)

