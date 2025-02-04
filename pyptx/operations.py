import logging
from .ptx_module import load_ptx_module, PTXModuleError

def matmul(gpu, optimized=False):
    try:
        module_name = "matmul_optimized" if optimized else "matmul"
        load_ptx_module(module_name)
    except PTXModuleError as e:
        logging.error(e)
        return
    logging.info(f"🚀 Running matmul on {'Optimized ' if optimized else ''}GPU {gpu.gpu_id}")
    # ...perform matrix multiplication...

def conv2d(gpu):
    try:
        load_ptx_module("conv2d")
    except PTXModuleError as e:
        logging.error(e)
        return
    logging.info(f"🚀 Running conv2d on GPU {gpu.gpu_id}")
    # ...perform convolution...

def weight_update():
    # Assuming weight update might be CPU bound or use a different module:
    logging.info("🚀 PyPTX Weight Update Success!")
    # ...perform weight update...
