import logging
import numpy as np
import os
import logging as logger
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests  # Import the requests library


# Add project root to path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from final.wrapper import PyPTXWrapper
from final.syntaxstructure import PyPTXSyntax
from final.pyptx_compiler import pyptx_compile
from final.Grapher import Grapher

def download_file(url, filename):
    """Downloads a file from a URL if it doesn't exist."""
    if not os.path.exists(filename):
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(filename, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {filename}")
    else:
        logger.info(f"{filename} already exists, skipping download")

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RealUseCase")

    # Load and preprocess Tiny Shakespeare dataset
    file_path = 'tinyshakespeare.txt'  # Replace with the actual path to the file
    file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # Updated valid URL

    download_file(file_url, file_path)  # Download the file if it doesn't exist

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}

    # Create input/output sequences
    sequence_length = 100
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - sequence_length, step):
        sentences.append(text[i: i + sequence_length])
        next_chars.append(text[i + sequence_length])

    X = np.zeros((len(sentences), sequence_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sentences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    # Define a character-level model using the DSL with longer training epochs
    model_code = f"""
    @model
    def shakespeare_net():
        layer("lstm", units=128, input_shape=({sequence_length}, {vocab_size}))
        layer("dense", input=128, output={vocab_size}, activation="softmax")
        train(epochs=50, learning_rate=0.01)
    """

    # Create wrapper and compile the high-level model description
    wrapper = PyPTXWrapper(model_code)
    syntax = PyPTXSyntax()
    syntax.parse(model_code)
    compiled_model = syntax.compile()
    logger.info(f"Compiled model:\n{compiled_model}")

    # Update the PTX code with real NN layer instructions (no TensorFlow/Keras)
    ptx_code = f"""
    .entry shakespeare_net {{
        .param .u64 lstm1, type="lstm", units=128, input_shape=({sequence_length}, {vocab_size});
        .param .u64 dense1, type="dense", input=128, output={vocab_size}, activation="softmax";
        .train_model(epochs=50, learning_rate=0.01);
    }}
    """
    compiled_ptx = pyptx_compile(ptx_code)
    logger.info(f"Compiled PTX code:\n{compiled_ptx.decode('utf-8')}")

    # Build a computation graph simulation
    graph = Grapher()
    operations = ["lstm", "dense"]
    for op in operations:
        graph.add_operation(op)
    logger.info("Executing graph operations...")
    graph.execute()

    # Simulate PTX training on Tiny Shakespeare text data for many epochs with 3D visualization.
    logger.info("Starting PTX training simulation on text data for 500 epochs...")
    losses = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Random Noise')
    ax.set_zlabel('Loss')
    epochs = 200
    for epoch in range(1, epochs + 1):
        loss = (1.0 / epoch) + np.random.rand() * 0.05  # simulating loss
        losses.append(loss)
        noise = np.random.rand()  # arbitrary value for y-axis
        ax.scatter(epoch, noise, loss, c='b', marker='o')
        plt.pause(0.01)
        logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    logger.info("PTX training simulation complete.")
    plt.title('3D Training Loss Visualization')
    plt.show()

    # Instead of uniform randomness, compute character probabilities from the training text
    char_counts = np.array([text.count(c) for c in chars])
    char_probs = char_counts / char_counts.sum()

    logger.info("Generating sample text from trained PTX model simulation (using character frequency probabilities)...")
    seed_text = sentences[0]
    generated = seed_text
    current_sequence = seed_text[-sequence_length:]
    for i in range(200):
        next_char = np.random.choice(chars, p=char_probs)
        generated += next_char
        current_sequence = current_sequence[1:] + next_char
    logger.info("Sample generated text:")
    logger.info(generated)

if __name__ == "__main__":
    main()
