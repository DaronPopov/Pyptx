try:
    from pyptx.compilation import pyptx_compile
    from pyptx.syntaxstructure import PyPTXSyntax
    from pyptx.wrapper import PyPTXWrapper
    from pyptx.ml_framework import PyPTXModel, model, layer, train  # Added train
    from pyptx.tensor_graph import PyPTXExecutionGraph
    from pyptx.update import pyptx_weight_update
    from pyptx.optimizer import PyPTXOptimizer
    from pyptx.gpu_manager import PyPTXGPUManager  # Adjusted import (was gpu_manager)
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")

__all__ = [
    "pyptx_compile",
    "PyPTXSyntax",
    "PyPTXWrapper",
    "PyPTXModel",
    "PyPTXExecutionGraph",
    "pyptx_weight_update",
    "PyPTXOptimizer",
    "PyPTXGPUManager",  # Exposed GPUManager from gpu_manager module
    "model",
    "layer",
    "train",  # Added train to __all__
]

# Note: Additional imports and __all__ entries should be added
# as their corresponding modules are implemented

if __name__ == "__main__":
    print("This module is not meant to be executed directly. Use it as a library.")
