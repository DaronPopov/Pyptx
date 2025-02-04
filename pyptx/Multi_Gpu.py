import platform
import ctypes
import subprocess
from .ptx_utils import ptx_loader

class PyPTXMultiGPU:
    @staticmethod
    def detect_gpus():
        system = platform.system().lower()
        if system == 'windows':
            try:
                import wmi
                return len([gpu for gpu in wmi.WMI().Win32_VideoController()])
            except:
                return 0
        elif system == 'linux':
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi', '-L'])
                return len(nvidia_smi.decode().strip().split('\n'))
            except:
                return 0
        elif system == 'darwin':
            try:
                metal_devices = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'])
                return 1 if b'Metal' in metal_devices else 0
            except:
                return 0
        return 0

    def __init__(self, num_gpus=1):
        if not ptx_loader.is_available():
            raise RuntimeError("NVIDIA driver not found")
            
        device_count = ptx_loader.get_device_count()
        if device_count == 0:
            raise RuntimeError("No NVIDIA GPUs detected")
            
        self.available_gpus = min(device_count, self.detect_gpus())
        self.num_gpus = min(num_gpus, self.available_gpus)
        
        # Store device handles and modules
        self.devices = []
        self.contexts = []
        self.modules = {}
        
        # Initialize devices with error checking
        for i in range(self.num_gpus):
            try:
                device = ctypes.c_int()
                context = ctypes.c_void_p()
                
                ptx_loader.check_error(
                    ptx_loader.nvidia_driver.cuDeviceGet(ctypes.byref(device), i)
                )
                ptx_loader.check_error(
                    ptx_loader.nvidia_driver.cuCtxCreate_v2(ctypes.byref(context), 0, device)
                )
                
                self.devices.append(device)
                self.contexts.append(context)
                print(f"🔥 Initialized GPU {i}")
            except Exception as e:
                print(f"Failed to initialize GPU {i}: {e}")
                continue
            
    def load_ptx_module(self, ptx_code, name="default"):
        """Load PTX module on all GPUs"""
        self.modules[name] = []
        for i, context in enumerate(self.contexts):
            ptx_loader.nvidia_driver.cuCtxSetCurrent(context)
            module = ptx_loader.load_ptx(ptx_code)
            self.modules[name].append(module)
            
    def execute_kernel(self, module_name, kernel_name, grid, block, args):
        """Execute PTX kernel across GPUs"""
        results = []
        for i, context in enumerate(self.contexts):
            ptx_loader.nvidia_driver.cuCtxSetCurrent(context)
            if module_name in self.modules:
                module = self.modules[module_name][i]
                kernel = ptx_loader.get_function(module, kernel_name)
                
                # Launch kernel
                ptx_loader.nvidia_driver.cuLaunchKernel(
                    kernel,
                    grid[0], grid[1], grid[2],  # Grid dimensions
                    block[0], block[1], block[2],  # Block dimensions
                    0, None,  # Shared memory and stream
                    args, None  # Arguments and extra
                )
                ptx_loader.nvidia_driver.cuCtxSynchronize()
                
        return results

    def get_memory_info(self):
        """Get memory information for all GPUs"""
        memory_info = []
        
        for i, context in enumerate(self.contexts):
            ptx_loader.nvidia_driver.cuCtxSetCurrent(context)
            
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            ptx_loader.nvidia_driver.cuMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            
            memory_info.append({
                'device': i,
                'free': free.value,
                'total': total.value,
                'used': total.value - free.value
            })
        
        return memory_info

    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'contexts'):
            for context in self.contexts:
                try:
                    ptx_loader.nvidia_driver.cuCtxDestroy_v2(context)
                except:
                    pass

# Initialize only if PTX is available
if ptx_loader.is_available():
    device_count = ptx_loader.get_device_count()
    if device_count > 0:
        print(f"🚀 Found {device_count} GPUs!")
        multi_gpu = PyPTXMultiGPU(device_count)
