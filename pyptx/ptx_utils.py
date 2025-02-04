import ctypes
import os
import platform

class PTXLoader:
    _instance = None
    nvidia_driver = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PTXLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        system = platform.system().lower()
        try:
            if system == 'windows':
                self.nvidia_driver = ctypes.WinDLL("nvcuda.dll")
            elif system == 'linux':
                self.nvidia_driver = ctypes.CDLL("libcuda.so.1")
            elif system == 'darwin':
                self.nvidia_driver = ctypes.CDLL("/usr/local/cuda/lib/libcuda.dylib")
            
            if self.nvidia_driver:
                self.nvidia_driver.cuInit(0)
                # Set function prototypes for better error handling
                self.nvidia_driver.cuDeviceGetCount.restype = int
                self.nvidia_driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                self.nvidia_driver.cuModuleLoadData.restype = int
                self.nvidia_driver.cuModuleGetFunction.restype = int
                
        except Exception as e:
            print(f"Warning: NVIDIA driver initialization failed: {e}")
            self.nvidia_driver = None
    
    def get_device_count(self):
        """Get number of available NVIDIA GPUs"""
        if not self.is_available():
            return 0
        
        count = ctypes.c_int(0)
        try:
            status = self.nvidia_driver.cuDeviceGetCount(ctypes.byref(count))
            if status == 0:
                return count.value
        except Exception as e:
            print(f"Error getting device count: {e}")
        return 0
    
    def is_available(self):
        return self.nvidia_driver is not None
        
    def load_ptx(self, ptx_code):
        """Load PTX code directly via driver"""
        if not self.is_available():
            raise RuntimeError("NVIDIA driver not available")
            
        ptx_source = ctypes.c_char_p(ptx_code.encode('utf-8'))
        module = ctypes.c_void_p()
        
        # Create module from PTX
        status = self.nvidia_driver.cuModuleLoadData(ctypes.byref(module), ptx_source)
        if status != 0:
            raise RuntimeError(f"Failed to load PTX module: {status}")
            
        return module
        
    def get_function(self, module, func_name):
        """Get function handle from PTX module"""
        function = ctypes.c_void_p()
        name = ctypes.c_char_p(func_name.encode('utf-8'))
        
        status = self.nvidia_driver.cuModuleGetFunction(
            ctypes.byref(function),
            module,
            name
        )
        if status != 0:
            raise RuntimeError(f"Failed to get function {func_name}: {status}")
            
        return function

    def check_error(self, status):
        """Universal error checking"""
        if status != 0:
            error_str = ctypes.c_char_p()
            self.nvidia_driver.cuGetErrorString(status, ctypes.byref(error_str))
            raise RuntimeError(f"NVIDIA Driver Error {status}: {error_str.value.decode()}")

    def allocate_memory(self, size):
        """Allocate device memory"""
        ptr = ctypes.c_void_p()
        status = self.nvidia_driver.cuMemAlloc(ctypes.byref(ptr), size)
        if status != 0:
            raise RuntimeError(f"Memory allocation failed: {status}")
        return ptr
    
    def copy_to_device(self, host_ptr, device_ptr, size):
        """Copy memory from host to device"""
        status = self.nvidia_driver.cuMemcpyHtoD(device_ptr, host_ptr, size)
        if status != 0:
            raise RuntimeError(f"Memory copy H2D failed: {status}")
            
    def copy_to_host(self, device_ptr, host_ptr, size):
        """Copy memory from device to host"""
        status = self.nvidia_driver.cuMemcpyDtoH(host_ptr, device_ptr, size)
        if status != 0:
            raise RuntimeError(f"Memory copy D2H failed: {status}")

ptx_loader = PTXLoader()
