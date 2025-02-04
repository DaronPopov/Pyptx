import pyptx
import numpy as np

# Generate random data with a pattern:
# The target is 1 if the sum of features is greater than a threshold, else 0.
num_samples = 10000
num_features = 640
X = np.random.rand(num_samples, num_features).astype(np.float32)
threshold = 8.0
y = (X.sum(axis=1) > threshold).astype(np.float32).reshape(-1, 1)

@pyptx.model
def neural_net():
    # Input layer with defined input shape
    pyptx.layer("dense", input_shape=(num_features,), units=32, activation="relu")
    # Hidden layer
    pyptx.layer("dense", units=16, activation="relu")
    # Output layer for binary classification
    pyptx.layer("dense", units=1, activation="sigmoid")
    
    # Train the model
    pyptx.train(
        X=X,
        y=y,
        epochs=200,
        batch_size=640,
        learning_rate=0.005
    )        
    return "Training completed!"

if __name__ == "__main__":
    result = neural_net()
    print(result)
