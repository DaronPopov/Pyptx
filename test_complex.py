import pyptx
import numpy as np
import platform
import sys

def check_platform_compatibility():
    system = platform.system().lower()
    if system not in ['windows', 'linux', 'darwin']:
        raise RuntimeError(f"Unsupported platform: {system}")
    return system

def test_wrapper_and_programmable_syntax():
    platform_type = check_platform_compatibility()
    code = """
@model
def neural_net():
    layer("dense", input=16, output=16, activation="relu")
    layer("conv2d", filters=3, kernel_size=3)
    train(epochs=10, learning_rate=0.01)
"""
    print("=== Testing PyPTXWrapper ===")
    pyptx.PyPTXWrapper(code).execute()

    print("\n=== Testing ProgrammableSyntax ===")
    from pyptx.programmable_syntax import ProgrammableSyntax
    ps = ProgrammableSyntax(code)
    compiled = ps.prepare()
    print("🚀 Compiled Code:\n", compiled)
    ps.execute()

def test_tensor_graph():
    print("\n=== Testing Tensor Graph Execution ===")
    from pyptx.tensor_graph import PyPTXExecutionGraph
    try:
        graph = PyPTXExecutionGraph()
        graph.add_operation("matmul", 0)
        graph.add_operation("conv2d", 1)
        graph.execute()
    except Exception as e:
        print(f"Error during tensor graph execution: {e}")

def test_self_learning():
    print("\n=== Testing Self Learning Module ===")
    from pyptx.pyptx_self_learning import PyPTXSelfLearning
    model = PyPTXSelfLearning()
    model.add_layer("dense")
    model.train()

def test_multi_gpu():
    print("\n=== Testing Multi GPU Execution ===")
    from pyptx.Multi_Gpu import PyPTXMultiGPU
    try:
        available_gpus = PyPTXMultiGPU.detect_gpus()
        if not available_gpus:
            print("No GPUs detected, skipping multi-GPU test")
            return
        mgpu = PyPTXMultiGPU(min(2, len(available_gpus)))
        mgpu.execute()
    except Exception as e:
        print(f"Error during multi-GPU execution: {e}")

def test_backprop():
    print("\n=== Testing Backpropagation ===")
    from pyptx.backprop import pyptx_backprop
    # Create a dummy model with layers
    class DummyLayer:
        pass
    class DummyModel:
        def __init__(self):
            self.layers = [DummyLayer(), DummyLayer()]
    dummy_model = DummyModel()
    loss = 0.5
    gradients = np.ones((10, 10), dtype=np.float32)
    updated_gradients = pyptx_backprop(dummy_model, loss, gradients)
    print("Updated Gradients:", updated_gradients)

def test_auto_train():
    print("\n=== Testing Auto Train ===")
    from pyptx.auto_train import PyPTXAutoTrain
    at = PyPTXAutoTrain()
    dummy_data = None  # Replace with actual training data as needed.
    at.train(dummy_data)
    at.evaluate(dummy_data)

if __name__ == "__main__":
    test_wrapper_and_programmable_syntax()
    test_tensor_graph()
    test_self_learning()
    test_multi_gpu()
    test_backprop()
    test_auto_train()
