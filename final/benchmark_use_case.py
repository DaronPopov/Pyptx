import logging
import time
import numpy as np
import os
import sys
import torch

# Add project root to path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from final.wrapper import PyPTXWrapper
from final.syntaxstructure import PyPTXSyntax
from final.pyptx_compiler import pyptx_compile
from final.Grapher import Grapher

logger = logging.getLogger("BenchmarkUseCase")
logger.setLevel(logging.INFO)

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("BenchmarkUseCase")
    
    # Define a heavy model using the DSL
    model_code = """
    @model
    def heavy_net():
        layer("conv2d", filters=64, kernel_size=3, activation="relu")
        layer("maxpool", pool_size=2)
        layer("conv2d", filters=128, kernel_size=3, activation="relu")
        layer("maxpool", pool_size=2)
        layer("flatten")
        layer("dense", input=128*7*7, output=512, activation="relu")
        layer("dropout", rate=0.5)
        layer("dense", input=512, output=10, activation="softmax")
        train(epochs=50, learning_rate=0.0002)
    """

    ptx_code = """
    .version 7.3
    .target sm_70
    .address_size 64

    // ... (rest of your PTX code) ...
    """

    logger.info("Starting DSL parsing and compilation benchmark...")
    start = time.time()
    syntax = PyPTXSyntax()
    syntax.parse(model_code)
    compiled_model = syntax.compile()
    dsl_time = time.time() - start
    logger.info(f"DSL compilation took {dsl_time:.4f} sec")
    logger.debug(f"Compiled model:\n{compiled_model}")

    logger.info("Starting PTX compilation benchmark...")
    start = time.time()
    compiled_ptx = pyptx_compile(ptx_code)
    ptx_time = time.time() - start
    logger.info(f"PTX compilation took {ptx_time:.4f} sec")

    # Benchmark graph creation and execution
    graph = Grapher()
    operations = ["conv2d", "maxpool", "conv2d", "maxpool", "flatten", "dense", "dropout", "dense"]
    for op in operations:
        graph.add_operation(op)
    start = time.time()
    graph.execute()
    graph_time = time.time() - start
    logger.info(f"Graph execution took {graph_time:.4f} sec")

    # Benchmark training simulation over a larger dummy dataset with compute-intensive operations
    # Remove the previous dummy loop for training simulation
    # New compute-intensive training simulation:
    X = np.random.randn(2000, 1000)   # large input matrix
    W = np.random.randn(1000, 1000)     # weight matrix
    epochs = 50
    logger.info("Starting compute-intensive training simulation benchmark...")
    start = time.time()
    for epoch in range(1, epochs + 1):
        # Perform several heavy matrix multiplications
        result = X.copy()
        for _ in range(5):  # chain of multiplications to simulate heavy compute load
            result = result.dot(W)
        # Compute a pseudo-loss value as normalized Frobenius norm
        loss = np.linalg.norm(result) / (result.shape[0]*result.shape[1])
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    training_time = time.time() - start
    logger.info(f"Compute-intensive training simulation over {epochs} epochs took {training_time:.4f} sec")

def benchmark_fft_torch(size, runs=10):
    """Benchmark FFT operation using NumPy and PyTorch (CUDA if available)."""
    logger.info(f"Benchmarking FFT (size={size})...")

    # NumPy FFT
    numpy_times = []
    for _ in range(runs):
        a_np = np.random.randn(size) + 1j * np.random.randn(size)
        start_time = time.time()
        np.fft.fft(a_np)
        numpy_times.append(time.time() - start_time)
    numpy_avg_time = np.mean(numpy_times)
    logger.info(f"[NumPy] FFT (size={size}) average time: {numpy_avg_time:.6f} sec")

    # PyTorch FFT
    torch_times = []
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU.")
    
    try:
        for _ in range(runs):
            a_torch = torch.randn(size, dtype=torch.complex64, device=device)
            start_time = time.time()
            torch.fft.fft(a_torch)
            if device == torch.device("cuda"):
                torch.cuda.synchronize()  # Ensure CUDA operations are finished
            torch_times.append(time.time() - start_time)
        torch_avg_time = np.mean(torch_times)
        logger.info(f"[PyTorch] FFT (size={size}) average time: {torch_avg_time:.6f} sec")
    except Exception as e:
        logger.error(f"Error during PyTorch FFT benchmark: {e}")

if __name__ == "__main__":
    main()
    benchmark_fft_torch(2**20)  # Example: 1M element FFT
