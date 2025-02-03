import pyptx
# ...existing code...

# High-Level Code to be tested
code = """
@model
def neural_net():
    layer("dense", input=16, output=16, activation="relu")
    layer("conv2d", filters=3, kernel_size=3)
    train(epochs=10, learning_rate=0.01)
"""

# Test using PyPTXWrapper
print("=== Testing PyPTXWrapper ===")
pyptx.PyPTXWrapper(code).execute()

# Test using ProgrammableSyntax interface
print("\n=== Testing ProgrammableSyntax ===")
from pyptx.programmable_syntax import ProgrammableSyntax
ps = ProgrammableSyntax(code)
compiled = ps.prepare()
print("🚀 Compiled Code:\n", compiled)
ps.execute()
