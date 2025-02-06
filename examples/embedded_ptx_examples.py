import numpy as np
import pyptx  # Imports model, layer, train, etc.
from final.pyptx_compiler import pyptx_compile

# Example: Model with Embedded PTX Script
@pyptx.model
def example_1():
    # Embedded PTX script showing low-level instructions
    embedded_ptx = """
    .entry embedded_model {
        .param .u64 dense1, type="dense", input=32, output=64, activation="relu";
        .param .u64 dense2, type="dense", input=64, output=10, activation="softmax";
        train_model(epochs=20, learning_rate=0.005);
    }
    """
    print("=== Embedded PTX Script ===")
    print(embedded_ptx)
    print("=== End of Embedded Script ===")
    
    # High-level DSL instructions
    pyptx.layer("dense", input=32, units=64, activation="relu")
    pyptx.layer("dense", units=10, activation="softmax")
    # Here we simulate training using the DSL call
    pyptx.train(epochs=20, learning_rate=0.005)
    
    # Optionally compile the embedded PTX script to verify its integration
    compiled = pyptx_compile(embedded_ptx)
    print("Compiled Embedded PTX (as bytes):", compiled)
    return embedded_ptx

# Additional examples can follow...
@pyptx.model
def example_2():
    """
    Another example demonstrating a different embedded script and model definition.
    """
    embedded_ptx = """
    .entry another_model {
        .param .u64 conv1, type="conv2d", filters=16, kernel_size=3, activation="relu";
        .param .u64 dense, type="dense", input=128, output=10, activation="softmax";
        train_model(epochs=15, learning_rate=0.01);
    }
    """
    print("=== Another Embedded PTX Script ===")
    print(embedded_ptx)
    print("=== End of Embedded Script ===")
    
    # DSL instructions corresponding to the embedded script
    pyptx.layer("conv2d", input=(28,28,1), filters=16, kernel_size=3, activation="relu")
    pyptx.layer("dense", units=10, activation="softmax")
    pyptx.train(epochs=15, learning_rate=0.01)
    
    compiled = pyptx_compile(embedded_ptx)
    print("Compiled Another Embedded PTX (as bytes):", compiled)
    return embedded_ptx

if __name__ == "__main__":
    # Run the embedded model example
    print("Running Embedded PTX Model Example:")
    example_1()
    
    # Run another example
    print("\nRunning Another Embedded PTX Example:")
    example_2()
