import ctypes
import numpy as np
from functools import wraps
from .backprop import pyptx_backprop
from pyptx.callbacks import BaseCallback
from layers import get_layer_impl
from pyptx_compiler import pyptx_compile

# Load NVIDIA driver
nvcuda = ctypes.WinDLL("nvcuda.dll")

# Initialize CUDA manually
nvcuda.cuInit(0)

# Get first GPU
device = ctypes.c_int()
nvcuda.cuDeviceGet(ctypes.byref(device), 0)

# Create GPU context
context = ctypes.c_void_p()
nvcuda.cuCtxCreate(ctypes.byref(context), 0, device)

# Define PyPTX Compile Function
from pyptx_compiler import pyptx_compile


# Global state
_current_model = None

class PyPTXModel:
    def __init__(self):
        self.layers = []
        self.compiled = False
        
    def add_layer(self, layer_type, **kwargs):
        self.layers.append({
            'type': layer_type,
            'params': kwargs,
            'weights': None,
            'bias': None
        })
        
    def compile(self):
        self._initialize_weights()
        self.compiled = True
        
    def _initialize_weights(self):
        for i, layer in enumerate(self.layers):
            if layer['type'] == 'dense':
                input_dim = (layer['params']['input_shape'][0] 
                           if i == 0 and 'input_shape' in layer['params'] 
                           else self.layers[i-1]['params']['units'])
                units = layer['params']['units']
                layer['weights'] = np.random.randn(input_dim, units) * 0.01
                layer['bias'] = np.zeros((1, units))

def model(func):
    """Model decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _current_model
        _current_model = PyPTXModel()
        result = func(*args, **kwargs)
        return _current_model
    return wrapper

def layer(layer_type, **kwargs):
    """Layer addition function"""
    global _current_model
    if _current_model is None:
        raise RuntimeError("layer() can only be called within a @model decorated function")
    _current_model.add_layer(layer_type, **kwargs)

def train(X, y, epochs=10, batch_size=32, learning_rate=0.01):
    """Training function"""
    global _current_model
    if _current_model is None:
        raise RuntimeError("train() can only be called within a @model decorated function")
    
    if not _current_model.compiled:
        _current_model.compile()
    
    num_samples = len(X)
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_X = X_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]
            
            # Forward pass with activation tracking
            current_batch = batch_X
            layer_inputs = []  # Store inputs for each layer
            for layer in _current_model.layers:
                if layer['type'] == 'dense':
                    layer_inputs.append(current_batch)  # Store input before transform
                    current_batch = np.dot(current_batch, layer['weights']) + layer['bias']
                    if layer['params'].get('activation') == 'relu':
                        current_batch = np.maximum(0, current_batch)
                    elif layer['params'].get('activation') == 'sigmoid':
                        current_batch = 1 / (1 + np.exp(-current_batch))
            
            # Compute loss
            loss = np.mean((current_batch - batch_y) ** 2)
            total_loss += loss
            
            # Backpropagation
            gradients = pyptx_backprop(_current_model, loss, current_batch - batch_y)
            
            # Update weights using stored layer inputs
            for j, layer in enumerate(_current_model.layers):
                if layer['type'] == 'dense':
                    layer_input = layer_inputs[j]
                    # Calculate weight gradients using current layer's input
                    weight_grad = np.dot(layer_input.T, gradients[j])
                    bias_grad = np.sum(gradients[j], axis=0, keepdims=True)
                    
                    # Update weights and biases
                    layer['weights'] -= learning_rate * weight_grad
                    layer['bias'] -= learning_rate * bias_grad
        
        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Export the functions
__all__ = ['model', 'layer', 'train', 'PyPTXModel']

# Example usage (do not call PyPTXModel() without a function):
if __name__ == "__main__":
    @model
    def example_model():
        print("Executing example model")
    example_model()
