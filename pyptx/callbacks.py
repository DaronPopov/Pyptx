class BaseCallback:
    def on_epoch_begin(self, epoch, logs=None):
        pass
        
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        pass

class ModelCheckpoint(BaseCallback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf')
        
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        current = val_metrics[self.monitor]
        if self.save_best_only:
            if current < self.best:
                self.best = current
                # Save model weights
                
class EarlyStopping(BaseCallback):
    def __init__(self, monitor='val_loss', patience=3):
        self.monitor = monitor
        self.patience = patience
        self.best = float('inf')
        self.wait = 0
        
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        current = val_metrics[self.monitor]
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True  # Stop training
