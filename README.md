# PyPTX Framework


A make shift implementation of cuda from scratch in python to allow PTX (parrelel thread execution) code to run via embedded assembley inside of python bypassing cuda. Direct GPU interface

 HEAD
A high-performance machine learning framework that directly leverages NVIDIA's PTX (Parallel Thread Execution) for CUDA acceleration. PyPTX provides a PyTorch-like interface while allowing low-level GPU optimization.
=======
A high-performance machine learning framework that directly leverages NVIDIA's PTX (Parallel Thread Execution) for CUDA acceleration. PyPTX provides a PyTorch-like interface while allowing low-level GPU optimization. Very buggy at the moment but i am devoloping it so any pr's are helpful. the compiler is buggy at the moment


## 🚀 Features

- Direct PTX code generation for CUDA operations
- PyTorch-like syntax for easy model creation
- Multi-GPU support with automatic workload distribution
- Self-learning capability with dynamic weight updates
- Customizable neural network layers
- Automatic gradient computation
- Windows support with CUDA integration

## 📦 Installation

```bash
pip install git+https://github.com/DaronPopov/pyptx.git
```

Requirements:
- Python 3.7+
- NVIDIA GPU with CUDA support
- Windows operating system
- CUDA Toolkit installed

## 🔥 Quick Start

```python
import pyptx
import numpy as np

@pyptx.model
def neural_net():
    pyptx.layer("dense", units=32, input_shape=(16,), activation="relu")
    pyptx.layer("dense", units=1)
    
    # Generate sample data
    X = np.random.randn(100, 16)
    y = np.random.randn(100, 1)
    
    # Train the model
    pyptx.train(X, y, epochs=10, batch_size=32, learning_rate=0.01)

model = neural_net()
```

## 🎛 Architecture

- `ml_framework.py`: Core ML functionality with model definition and training
- `compiler.py`: PTX code generation and compilation
- `tensor_graph.py`: Tensor operations execution graph
- `multi_gpu.py`: Multi-GPU workload distribution
- `syntax.py`: High-level syntax parsing
- `wrapper.py`: Python-PTX interface layer

## 🔧 Advanced Usage

### Multi-GPU Training

```python
from pyptx.multi_gpu import PyPTXMultiGPU

# Automatically distributes workload across available GPUs
multi_gpu = PyPTXMultiGPU(num_gpus=2)
multi_gpu.execute()
```

### Custom Layer Definition

```python
@pyptx.model
def custom_model():
    pyptx.layer("dense", units=64, activation="relu")
    pyptx.layer("custom", operation="my_operation")
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

Open Source - See LICENSE file for details.

## ⭐ Show Your Support

If you find PyPTX useful, please star the repository!

## 📫 Contact

- GitHub: [@DaronPopov](https://github.com/DaronPopov)
