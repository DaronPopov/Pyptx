import pyptx
import numpy as np

def test_wrapper_and_programmable_syntax():
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
    graph = PyPTXExecutionGraph()
    graph.add_operation("matmul", 0)
    graph.add_operation("conv2d", 1)
    graph.execute()

def test_self_learning():
    print("\n=== Testing Self Learning Module ===")
    from pyptx.pyptx_self_learning import PyPTXSelfLearning
    model = PyPTXSelfLearning()
    model.add_layer("dense")
    model.train()

def test_multi_gpu():
    print("\n=== Testing Multi GPU Execution ===")
    from pyptx.Multi_Gpu import PyPTXMultiGPU
    # For testing, assume 2 GPUs; adjust based on available hardware.
    mgpu = PyPTXMultiGPU(2)
    mgpu.execute()

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
