class PyPTXMemoryOptimizer:
    def __init__(self):
        self.memory_map = {}

    def allocate_tensor(self, tensor_name, size):
        """Allocates memory efficiently"""
        self.memory_map[tensor_name] = size
        print(f"🔥 Allocated {size}MB for {tensor_name}")

    def deallocate_tensor(self, tensor_name):
        """Deallocates memory once a tensor is no longer needed"""
        if tensor_name in self.memory_map:
            print(f"💀 Deallocating {tensor_name}")
            del self.memory_map[tensor_name]

# Example Usage
mem_opt = PyPTXMemoryOptimizer()
mem_opt.allocate_tensor("activations", 1024)
mem_opt.deallocate_tensor("activations")
