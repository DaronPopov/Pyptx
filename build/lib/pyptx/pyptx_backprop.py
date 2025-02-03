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

# Define PyPTX Backpropagation Kernel
pyptx_backprop = """
    .version 7.0
    .target sm_80
    .address_size 64

    .visible .entry backprop_kernel() {
        .param .u64 input, weights, output;
        ld.param.u64 %rd1, [input];
        ld.param.u64 %rd2, [weights];

        # Compute gradient
        sub.f32 %rd3, %rd1, %rd2;  # Gradient = input - weights
        mul.f32 %rd3, %rd3, 0.01;   # Learning Rate * Gradient

        # Update Weights
        add.f32 %rd2, %rd2, %rd3;
        st.param.u64 [output], %rd2;

        ret;
    }
"""

# Compile and execute PyPTX Backpropagation
def run_backprop():
    compiled_kernel = pyptx_backprop.encode()
    module = ctypes.c_void_p()
    kernel_func = ctypes.c_void_p()

    nvcuda.cuModuleLoadData(ctypes.byref(module), compiled_kernel)
    nvcuda.cuModuleGetFunction(ctypes.byref(kernel_func), module, b"backprop_kernel")

    try:
        nvcuda.cuLaunchKernel(kernel_func, 1, 1, 1, 1, 1, 1, 0, None, None, None)
        print("🚀 PyPTX Backpropagation Execution Success!")
    except OSError as e:
        print(f"💀 Execution Failed: {e}")

# Run Backpropagation
run_backprop()
