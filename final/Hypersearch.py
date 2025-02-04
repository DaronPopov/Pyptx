class PyPTXHyperSearch:
    def __init__(self):
        self.hyperparameters = {
            "learning_rate": [0.001, 0.005, 0.01],
            "batch_size": [16, 32, 64],
            "dropout": [0.1, 0.2, 0.3]
        }

    def run_search(self):
        """Finds the best combination of hyperparameters"""
        best_config = None
        best_score = float("inf")

        for lr in self.hyperparameters["learning_rate"]:
            for batch in self.hyperparameters["batch_size"]:
                for dropout in self.hyperparameters["dropout"]:
                    score = self.evaluate_model(lr, batch, dropout)
                    if score < best_score:
                        best_score = score
                        best_config = (lr, batch, dropout)

        print(f"🚀 Best Hyperparameter Configuration: LR={best_config[0]}, Batch={best_config[1]}, Dropout={best_config[2]}")

    def evaluate_model(self, lr, batch, dropout):
        """Simulates evaluating a model with given hyperparameters"""
        return (lr * 1000) + (batch / 2) + (dropout * 100)

# Example Usage
hypersearch = PyPTXHyperSearch()
hypersearch.run_search()
