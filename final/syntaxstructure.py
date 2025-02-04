import re

class PyPTXSyntax:
    def __init__(self):
        self.model_code = []
        self.model_name = "neural_net"  # default name

    def parse(self, code):
        """Parses high-level PyPTX syntax into tensor execution instructions"""
        lines = code.strip().split("\n")
        for line in lines:
            if line.strip().startswith("@model"):
                # Extract model name from the next line
                for next_line in lines[lines.index(line)+1:]:
                    if "def" in next_line:
                        self.model_name = re.search(r'def\s+(\w+)', next_line).group(1)
                        break
                self.model_code.append(f".entry {self.model_name} {{")
            elif "layer" in line:
                layer_type, params = self.extract_params(line)
                self.model_code.append(f"    .param .u64 {layer_type}, {params};")
            elif "train" in line:
                epochs, lr = self.extract_train_params(line)
                self.model_code.append(f"    train_model(epochs={epochs}, lr={lr});")
        self.model_code.append("}")

    def extract_params(self, line):
        """Extracts layer parameters"""
        match = re.search(r'layer\("(.+?)", (.+)\)', line)
        if match:
            return match.group(1), match.group(2)
        return "", ""

    def extract_train_params(self, line):
        """Extracts training parameters"""
        match = re.search(r'train\(epochs=(\d+), learning_rate=(.+)\)', line)
        if match:
            return match.group(1), match.group(2)
        return "1", "0.01"

    def compile(self):
        """Compiles parsed code into PyPTX execution"""
        return "\n".join(self.model_code)

# Example Usage
syntax_parser = PyPTXSyntax()
high_level_code = """
@model
def neural_net():
    layer("dense", input=16, output=16, activation="relu")
    layer("conv2d", filters=3, kernel_size=3)
    train(epochs=10, learning_rate=0.01)
"""
syntax_parser.parse(high_level_code)
compiled_code = syntax_parser.compile()
print("🚀 Compiled PyPTX Code:\n", compiled_code)
