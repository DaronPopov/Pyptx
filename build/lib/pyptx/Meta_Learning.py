class PyPTXMetaLearning:
    def __init__(self):
        self.history = []

    def log_training(self, epoch, loss, accuracy):
        """Records model training history"""
        self.history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})

    def analyze_performance(self):
        """Adjusts hyperparameters based on past training"""
        if len(self.history) > 2:
            recent_loss = self.history[-1]["loss"]
            prev_loss = self.history[-2]["loss"]
            if recent_loss > prev_loss:
                print("⚡ Lowering learning rate to stabilize training.")
            else:
                print("🚀 Increasing learning rate for faster convergence.")

# Example Usage
meta = PyPTXMetaLearning()
meta.log_training(1, 0.4, 85.2)
meta.log_training(2, 0.35, 86.5)
meta.analyze_performance()
