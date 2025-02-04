from syntaxstructure import PyPTXSyntax
from pyptx_compiler import pyptx_compile
from Grapher import Grapher
import re

class PyPTXError(Exception):
    """Base exception for PyPTX errors"""
    pass

class PyPTXSyntaxError(PyPTXError):
    """Raised when syntax parsing fails"""
    pass

class PyPTXWrapper:
    def __init__(self, code=None):
        self.syntax = PyPTXSyntax()
        self.code = code
        self.grapher = Grapher()

    def validate_code(self):
        """Validate code structure before parsing"""
        if not self.code:
            raise PyPTXError("No code provided")
        
        # Check for basic structure
        if "@model" not in self.code:
            raise PyPTXSyntaxError("Missing @model decorator")
            
        # Validate layer definitions
        layer_pattern = r'layer\("(\w+)",\s*([^)]+)\)'
        for layer_match in re.finditer(layer_pattern, self.code):
            layer_type, params = layer_match.groups()
            if not params.strip():
                raise PyPTXSyntaxError(f"Invalid parameters for layer: {layer_type}")

    def execute(self):
        """Parses and runs high-level PyPTX code"""
        self.validate_code()
        self.syntax.parse(self.code)
        compiled_code = self.syntax.compile()
        compiled_bytes = pyptx_compile(compiled_code)
        print("🚀 PyPTX Execution Complete!")
        return compiled_bytes

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
