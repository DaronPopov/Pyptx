import psutil

class PyPTXGPUManager:
    def __init__(self):
        self.gpu_usage = {0: 0, 1: 0}  # Example for 2 GPUs

    def update_usage(self, gpu_id, load):
        """Updates GPU usage dynamically"""
        self.gpu_usage[gpu_id] += load

    def get_least_used_gpu(self):
        """Finds the least busy GPU"""
        return min(self.gpu_usage, key=self.gpu_usage.get)

    def balance_load(self):
        """Rebalances workloads if one GPU is overloaded"""
        max_gpu = max(self.gpu_usage, key=self.gpu_usage.get)
        min_gpu = min(self.gpu_usage, key=self.gpu_usage.get)
        if self.gpu_usage[max_gpu] - self.gpu_usage[min_gpu] > 2000:  # Arbitrary threshold
            print(f"⚡ Rebalancing load: Moving tasks from GPU {max_gpu} to GPU {min_gpu}")

# Example Usage
gpu_manager = PyPTXGPUManager()
gpu_manager.update_usage(0, 5000)
gpu_manager.update_usage(1, 7000)
gpu_manager.balance_load()
