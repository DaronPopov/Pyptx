from .syntaxstructure import PyPTXSyntax
from .compilation import pyptx_compile

class PyPTXWrapper:
    def __init__(self, code):
        self.syntax = PyPTXSyntax()
        self.code = code

    def execute(self):
        """Parses and runs high-level PyPTX code"""
        self.syntax.parse(self.code)
        compiled_code = self.syntax.compile()
        # Use pyptx_compile as a function to get compiled bytes.
        compiled_bytes = pyptx_compile(compiled_code)
        # Optionally, execute compiled_bytes or handle accordingly.
        print("🚀 PyPTX Execution Complete!")

# Example: Running High-Level PyPTX Code
code = """
@model
def neural_net():
    layer("dense", input=16, output=16, activation="relu")
    layer("conv2d", filters=3, kernel_size=3)
    train(epochs=10, learning_rate=0.01)
"""
pyptx = PyPTXWrapper(code)
pyptx.execute()
