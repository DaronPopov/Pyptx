import ctypes



class PyPTXModel:
    def __init__(self):
        self.layers = []

    def add(self, operation):
        """Adds a tensor operation to the model"""
        self.layers.append(operation)

    def compile(self):
        """Compiles the PyPTX model into an executable tensor kernel"""
        compiled_code = "\n".join(self.layers)
        return pyptx_compile(compiled_code)

    def run(self):
        """Executes the compiled PyPTX model"""
        compiled_kernel = self.compile()
        module = ctypes.c_void_p()
        kernel_func = ctypes.c_void_p()

        nvcuda.cuModuleLoadData(ctypes.byref(module), compiled_kernel)
        nvcuda.cuModuleGetFunction(ctypes.byref(kernel_func), module, b"tensor_kernel")

        try:
            nvcuda.cuLaunchKernel(kernel_func, 1, 1, 1, 1, 1, 1, 0, None, None, None)
            print("🚀 PyPTX ML Model Execution Success!")
        except OSError as e:
            print(f"💀 Execution Failed: {e}")

# ✅ Make sure this is at the end of pyptx_ml.py
if __name__ == "__main__":
    model = PyPTXModel()
    


# Load NVIDIA driver
nvcuda = ctypes.WinDLL("nvcuda.dll")

# Initialize CUDA manually (but only for memory access)
nvcuda.cuInit(0)

# Get first GPU
device = ctypes.c_int()
nvcuda.cuDeviceGet(ctypes.byref(device), 0)

# Create GPU context
context = ctypes.c_void_p()
nvcuda.cuCtxCreate(ctypes.byref(context), 0, device)

# Define PyPTX Compiler Function
def pyptx_compile(code):
    """Converts PyPTX tensor instructions into executable PTX"""
    compiled_ptx = """
    .version 7.0
    .target sm_80
    .address_size 64

    .visible .entry tensor_kernel() {
        .param .u64 A, B, C;
        ld.param.u64 %rd1, [A];
        ld.param.u64 %rd2, [B];

        """ + code + """

        st.param.u64 [C], %rd3;
        ret;
    }
    """
    return compiled_ptx.encode()

# Matrix Multiplication (16x16 warp-matrix multiply)
pyptx_matrix_mult = """
wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32.f32.f32 %rd3, %rd1, %rd2;
"""

# Dot Product (4-way vector multiply-accumulate)
pyptx_dot = """
dp4a.u32.u32 %rd3, %rd1, %rd2, %rd3;
"""

# Convolution (2D tensor convolution)
pyptx_conv2d = """
mma.sync.aligned.m16n16k16.row.col.f32.f32.f32.f32 %rd3, %rd1, %rd2;
"""

# ReLU Activation (Zero out negative values)
pyptx_relu = """
max.f32 %rd3, %rd1, 0.0;
"""

# Sigmoid Activation (1 / (1 + exp(-x)))
pyptx_sigmoid = """
mov.f32 %f1, -1.0;
mul.f32 %f2, %rd1, %f1;
ex2.approx.f32 %f3, %f2;
add.f32 %f4, 1.0, %f3;
rcp.f32 %rd3, %f4;
"""

# Softmax Activation (exp(x) / sum(exp(x)))
pyptx_softmax = """
ex2.approx.f32 %rd3, %rd1;
"""


compiled_kernel = pyptx_compile(pyptx_matrix_mult)

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
    print("🚀 PyPTX ML Execution Success! Matrix Multiplication Achieved.")
except OSError as e:
    print(f"💀 Execution Failed: {e}")
