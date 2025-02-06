# PyPTX
This is the alpha build so you will have to possibly fix some things to work for your use case but this is the lowest level gpu access through python
A questionabley high-performance machine learning framework that directly leverages NVIDIA's PTX (Parallel Thread Execution) through the nividia drivers to bypass the cuda annoyance. write direct to gpus with python and ptx assembley. PyPTX provides a PyTorch-like interface while allowing low-level GPU optimization. buggy but works with a little effort (:

## 🚀 Features

- Direct PTX code generation for CUDA operations
- PyTorch-like syntax for easy model creation: heavy emphasis on "like"
- Multi-GPU support with automatic workload distribution
- Self-learning capability with dynamic weight updates: possibly(; 
- Customizable neural network layers
- Automatic gradient computation


## 📦 Installation

You can install the package directly from GitHub:

```sh
pip install git+https://github.com/DaronPopov/pyptx.git
```

Requirements:
- Python 3.7+
- NVIDIA GPU (dont trip bout cuda or nun of all that)
- Windows operating system, Gods gift to humanity 


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
- `combiner.py`: PTX code generation and combination
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
- be annoying
- Suggest new problems
- Submit pull requests i dont care have fun do whatever i have the og build lol

## 📄 License

there is no license i dont care yall can rip this all you want 

## ⭐ Show Your Support

If you find PyPTX useful, please leave and dont use again just sit there and ruminate if cuda was useful ever?

## 📫 Contact

- GitHub: [@DaronPopov](https://github.com/DaronPopov)
