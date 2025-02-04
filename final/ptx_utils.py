import ctypes
import os
import platform
import logging

class PTXLoader:
    _instance = None
    nvidia_driver = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PTXLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize CUDA driver and create context"""
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
                
                # Create context
                device = ctypes.c_int()
                self.nvidia_driver.cuDeviceGet(ctypes.byref(device), 0)
                self.context = ctypes.c_void_p()
                self.nvidia_driver.cuCtxCreate(ctypes.byref(self.context), 0, device)
                
                # Set function prototypes
                self.nvidia_driver.cuMemAlloc_v2 = self.nvidia_driver.cuMemAlloc
                self.nvidia_driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
                self.nvidia_driver.cuMemAlloc_v2.restype = int
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"NVIDIA driver initialization failed: {e}")
            self.nvidia_driver = None
            self.context = None
    
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
        """Allocate device memory with proper error handling"""
        if not self.is_available():
            raise RuntimeError("NVIDIA driver not available")
            
        ptr = ctypes.c_void_p()
        try:
            status = self.nvidia_driver.cuMemAlloc_v2(ctypes.byref(ptr), size)
            if status != 0:
                if status == 201:  # CUDA_ERROR_OUT_OF_MEMORY
                    raise RuntimeError("GPU out of memory")
                raise RuntimeError(f"Memory allocation failed with status: {status}")
            return ptr
        except Exception as e:
            raise RuntimeError(f"Memory allocation failed: {str(e)}")
    
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
