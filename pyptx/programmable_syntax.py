from .syntax import PyPTXSyntax
from .wrapper import PyPTXWrapper

class ProgrammableSyntax:
    def __init__(self, code):
        self.code = code
        self.parser = PyPTXSyntax()

    def prepare(self):
        # Parse the high-level syntax and compile it
        self.parser.parse(self.code)
        return self.parser.compile()

    def execute(self):
        # Use the wrapper to execute the high-level code
        PyPTXWrapper(self.code).execute()
        
# ...existing code...
if __name__ == "__main__":
    high_level_code = """
    @model
    def neural_net():
        layer("dense", input=16, output=16, activation="relu")
        layer("conv2d", filters=3, kernel_size=3)
        train(epochs=10, learning_rate=0.01)
    """
    ps = ProgrammableSyntax(high_level_code)
    compiled = ps.prepare()
    print("🚀 Compiled Code:\n", compiled)
    ps.execute()
