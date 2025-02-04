class PyPTXDistributedTraining:
    def __init__(self, num_nodes, num_gpus_per_node):
        self.nodes = num_nodes
        self.gpus_per_node = num_gpus_per_node
        self.global_gpu_count = num_nodes * num_gpus_per_node
        self.gradient_shards = {}

    def distribute_gradients(self, tensor_name, size):
        """Splits gradients across multiple nodes for faster training"""
        shard_size = size // self.global_gpu_count
        self.gradient_shards[tensor_name] = [
            {"node": i // self.gpus_per_node, "gpu": i % self.gpus_per_node, "size": shard_size}
            for i in range(self.global_gpu_count)
        ]

    def execute_gradient_aggregation(self, tensor_name):
        """Aggregates gradients across all GPUs and nodes"""
        shards = self.gradient_shards.get(tensor_name, [])
        for shard in shards:
            print(f"🚀 Aggregating {tensor_name} gradients on Node {shard['node']}, GPU {shard['gpu']} (Shard Size: {shard['size']})")

# Example Usage
distributed_training = PyPTXDistributedTraining(num_nodes=4, num_gpus_per_node=8)
distributed_training.distribute_gradients("model_gradients", size=64000)
distributed_training.execute_gradient_aggregation("model_gradients")
