import logging
import numpy as np

logger = logging.getLogger(__name__)

class Grapher:
    def __init__(self):
        self.operations = []
        self.parameters = []  # Change to list to maintain order
        logger.debug("Initialized PyPTX Grapher")

    def add_operation(self, operation, parameters=None):
        """Adds an operation to the execution graph."""
        self.operations.append(operation)
        self.parameters.append(parameters)  # Will be None if no parameters
        logger.debug(f"Added operation: {operation}")

    def execute(self):
        """Executes all operations in the graph."""
        logger.info("Executing PyPTX graph operations...")
        current_output = None
        
        for i, (op, params) in enumerate(zip(self.operations, self.parameters)):
            logger.info(f"Executing operation: {op}")
            current_output = self._apply_operation(op, params, current_output)
        
        logger.info("Execution of PyPTX graph complete.")
        return current_output

    def _apply_operation(self, op_type, parameters, input_data):
        """Apply operation with parameters to input data"""
        try:
            if op_type.startswith('dense'):
                if parameters is not None and input_data is not None:
                    return np.dot(input_data, parameters)
                return parameters
            elif op_type == 'relu':
                return np.maximum(0, input_data) if input_data is not None else None
            return input_data
        except Exception as e:
            logger.error(f"Error executing operation {op_type}: {e}")
            return input_data

if __name__ == "__main__":
    graph = Grapher()
    graph.add_operation("matmul")
    graph.add_operation("conv2d")
    graph.execute()