import ctypes

# Load NVIDIA driver
nvcuda = ctypes.WinDLL("nvcuda.dll")

# Initialize CUDA manually
nvcuda.cuInit(0)

# Detect all available GPUs
num_gpus = ctypes.c_int()
nvcuda.cuDeviceGetCount(ctypes.byref(num_gpus))
print(f"🚀 Found {num_gpus.value} GPUs!")

# Create contexts for each GPU
contexts = []
for i in range(num_gpus.value):
    device = ctypes.c_int(i)
    context = ctypes.c_void_p()
    nvcuda.cuCtxCreate(ctypes.byref(context), 0, device)
    contexts.append(context)
    print(f"🔥 Created GPU Context for Device {i}")

# Function to assign workload to GPUs
def distribute_workload():
    for i, context in enumerate(contexts):
        nvcuda.cuCtxSetCurrent(context)  # Switch to the GPU context

        # Launch a tensor operation
        print(f"🚀 Executing on GPU {i}")
        # Add your tensor kernel execution here

distribute_workload()
