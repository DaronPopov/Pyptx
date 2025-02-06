import numpy as np
import pyptx  # Imports model, layer, train, etc.

# Example 1: Simple Dense Network
@pyptx.model
def dense_network_example():
    """
    A simple dense network with two fully connected layers.
    """
    pyptx.layer("dense", input=32, units=64, activation="relu")
    pyptx.layer("dense", units=10, activation="softmax")
    
    # Generate dummy data
    X = np.random.randn(100, 32)
    y = np.random.randn(100, 10)
    
    # Train the model
    pyptx.train(X, y, epochs=10, batch_size=16, learning_rate=0.01)
    return "Dense network training complete."

# Example 2: Convolutional Network
@pyptx.model
def cnn_example():
    """
    A simple CNN that uses a convolutional layer followed by a dense classification layer.
    """
    pyptx.layer("conv2d", input=(28, 28, 1), filters=32, kernel_size=3, activation="relu")
    pyptx.layer("dense", units=10, activation="softmax")
    
    # Generate dummy image data
    X = np.random.randn(100, 28, 28, 1)
    y = np.random.randn(100, 10)
    
    # Train the network
    pyptx.train(X, y, epochs=5, batch_size=10, learning_rate=0.005)
    return "CNN training complete."

# Example 3: LSTM-based Sequential Model
@pyptx.model
def lstm_example():
    """
    An example of an LSTM network for sequential data.
    """
    pyptx.layer("lstm", input_shape=(50, 20), units=128)
    pyptx.layer("dense", units=5, activation="softmax")
    
    # Generate dummy sequential data
    X = np.random.randn(50, 50, 20)
    y = np.random.randn(50, 5)
    
    # Train the LSTM model
    pyptx.train(X, y, epochs=8, batch_size=5, learning_rate=0.01)
    return "LSTM network training complete."

# Example 4: Custom Layer and Advanced Settings
@pyptx.model
def custom_layer_example():
    """
    An example that uses a custom layer operation.
    """
    pyptx.layer("dense", input=64, units=128, activation="relu")
    pyptx.layer("custom", operation="my_custom_op")
    pyptx.layer("dense", units=10, activation="softmax")
    
    # Generate custom input data
    X = np.random.randn(120, 64)
    y = np.random.randn(120, 10)
    
    # Train the custom network
    pyptx.train(X, y, epochs=12, batch_size=20, learning_rate=0.002)
    return "Custom layer network training complete."

# Example 5: Multi-GPU Simulation Example
@pyptx.model
def multi_gpu_example():
    """
    Demonstrates usage in a multi-GPU setting.
    """
    pyptx.layer("dense", input=100, units=256, activation="relu")
    pyptx.layer("dense", units=100, activation="softmax")
    
    # Simulate a multi-GPU scenario with dummy data
    X = np.random.randn(200, 100)
    y = np.random.randn(200, 100)
    
    # Train with multi-GPU settings simulated (actual handler in multi_gpu module)
    pyptx.train(X, y, epochs=15, batch_size=25, learning_rate=0.005)
    return "Multi-GPU simulation training complete."

if __name__ == "__main__":
    print(dense_network_example())
    print(cnn_example())
    print(lstm_example())
    print(custom_layer_example())
    print(multi_gpu_example())
