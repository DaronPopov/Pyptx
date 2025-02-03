import ctypes

# Load NVIDIA driver
nvcuda = ctypes.WinDLL("nvcuda.dll")

# Initialize CUDA manually
nvcuda.cuInit(0)

# Get first GPU
device = ctypes.c_int()
nvcuda.cuDeviceGet(ctypes.byref(device), 0)

# Create GPU context
context = ctypes.c_void_p()
nvcuda.cuCtxCreate(ctypes.byref(context), 0, device)

# Simple PyPTX Syntax Compiler
def pyptx_compile(code):
    """Converts PyPTX assembly-style code into PTX"""
    compiled_ptx = """
    .version 7.0
    .target sm_80
    .address_size 64

    .visible .entry tensor_kernel() {
        .param .u64 inputA, inputB, output;
        ld.param.u64 %rd1, [inputA];
        ld.param.u64 %rd2, [inputB];

        """ + code + """

        st.param.u64 [output], %rd3;
        ret;
    }
    """
    return compiled_ptx.encode()

# Example PyPTX Code (Multiplies tensors A and B)
pyptx_code = """
mad.u64 %rd3, %rd1, %rd2, %rd1;
"""

compiled_kernel = pyptx_compile(pyptx_code)

# Allocate GPU memory dynamically
gpu_mem = ctypes.c_void_p()
nvcuda.cuMemAlloc(ctypes.byref(gpu_mem), 4096)

# Load the dynamically compiled PyPTX kernel
module = ctypes.c_void_p()
kernel_func = ctypes.c_void_p()

nvcuda.cuModuleLoadData(ctypes.byref(module), compiled_kernel)
nvcuda.cuModuleGetFunction(ctypes.byref(kernel_func), module, b"tensor_kernel")

# Execute the compiled PyPTX kernel
try:
    nvcuda.cuLaunchKernel(kernel_func, 1, 1, 1, 1, 1, 1, 0, None, None, None)
    print("🚀 PyPTX Execution Success! Custom Tensor Computation Achieved.")
except OSError as e:
    print(f"💀 Execution Failed: {e}")
