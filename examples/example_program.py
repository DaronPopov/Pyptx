import numpy as np
import pyptx  # Imports model, layer, train and others

# Define a simple model using the PyPTX DSL
@pyptx.model
def demo_model():
    pyptx.layer("dense", input=16, units=32, activation="relu")
    pyptx.layer("dense", units=10, activation="softmax")
    
    # Generate dummy data for training
    X = np.random.randn(100, 16)
    y = np.random.randn(100, 10)
    
    # Train the model with specified parameters
    pyptx.train(X, y, epochs=20, batch_size=16, learning_rate=0.005)

if __name__ == "__main__":
    # Execute the model creation and training
    model_instance = demo_model()
    # ...existing code...
    print("Demo model executed using PyPTX DSL!")
