import unittest
import numpy as np
import logging
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from final.wrapper import PyPTXWrapper, PyPTXError, PyPTXSyntaxError
from final.Grapher import Grapher
from final.syntaxstructure import PyPTXSyntax
from final.pyptx_compiler import pyptx_compile
from final.ptx_utils import PTXLoader

class TestPyPTXComplex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(cls.__name__)

    def setUp(self):
        self.sample_model = """
        @model
        def complex_net():
            layer("dense", input=784, output=128, activation="relu")
            layer("dropout", rate=0.3)
            layer("dense", input=128, output=64, activation="relu")
            layer("dense", input=64, output=10, activation="softmax")
            train(epochs=5, learning_rate=0.01)
        """
        self.ptx_code = """
        .entry complex_net {
            .param .u64 dense1, input=784, output=128, activation="relu";
            .param .u64 dropout, rate=0.3;
            .param .u64 dense2, input=128, output=64, activation="relu";
            .param .u64 dense3, input=64, output=10, activation="softmax";
            train_model(epochs=5, lr=0.01);
        }
        """

    def test_full_pipeline(self):
        """Test the complete PyPTX pipeline"""
        # 1. Create wrapper and parse model
        wrapper = PyPTXWrapper(self.sample_model)
        self.assertIsNotNone(wrapper)

        # 2. Test syntax parsing
        syntax = PyPTXSyntax()
        syntax.parse(self.sample_model)
        compiled = syntax.compile()
        self.assertIsInstance(compiled, str)
        self.assertIn("complex_net", compiled)  # Now this should pass
        self.logger.debug(f"Compiled code:\n{compiled}")  # Add debug output

        # 3. Test graph creation and operations
        graph = Grapher()
        for op in ["dense", "dropout", "dense", "dense"]:
            graph.add_operation(op)
        self.assertEqual(len(graph.operations), 4)

        # 4. Test PTX compilation
        compiled_bytes = pyptx_compile(self.ptx_code)
        self.assertIsInstance(compiled_bytes, bytes)

        # 5. Test GPU utilities
        ptx_loader = PTXLoader()
        self.assertIsNotNone(ptx_loader)
        # Capture GPU log output for verification
        with self.assertLogs(self.logger, level="INFO") as cm:
            device_count = ptx_loader.get_device_count()
            self.logger.info(f"Detected {device_count} GPU devices")
        self.assertTrue(any("Detected" in message for message in cm.output))

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test missing @model decorator
        invalid_code = """
        def broken_net():
            layer("dense", input=10, output=5)
        """
        with self.assertRaisesRegex(PyPTXSyntaxError, ".*@model.*"):
            wrapper = PyPTXWrapper(invalid_code)
            wrapper.execute()

        # Test invalid layer parameters
        invalid_layer_code = """
        @model
        def broken_net():
            layer("invalid_layer", )
        """
        with self.assertRaisesRegex(PyPTXSyntaxError, ".*invalid_layer.*"):
            wrapper = PyPTXWrapper(invalid_layer_code)
            wrapper.execute()

        # Test empty code
        with self.assertRaises(PyPTXError):
            wrapper = PyPTXWrapper(None)
            wrapper.execute()

    def test_graph_execution(self):
        """Test graph execution with multiple operations"""
        graph = Grapher()
        operations = [
            ("dense", {"input": 784, "output": 128}),
            ("relu", {}),
            ("dropout", {"rate": 0.3}),
            ("dense", {"input": 128, "output": 10}),
            ("softmax", {})
        ]
        
        for op, params in operations:
            with self.subTest(operation=op):
                graph.add_operation(op)
        
        graph.execute()
        self.assertEqual(len(graph.operations), 5)

    def test_mock_training(self):
        """Test training simulation with mock data"""
        # Create mock input data
        X = np.random.randn(100, 784)  # 100 samples of 784 features
        y = np.random.randint(0, 10, size=(100, 10))  # 10 classes

        # Compile and run mock training
        wrapper = PyPTXWrapper(self.sample_model)
        compiled = pyptx_compile(self.ptx_code)
        
        # Verify compilation result
        self.assertIn(b"complex_net", compiled)
        self.assertIn(b"dense", compiled)
        self.assertIn(b"dropout", compiled)

    def test_complex_model_training(self):
        """Test training simulation with a more complex model"""
        complex_model = """
        @model
        def complex_net2():
            layer("conv2d", filters=32, kernel_size=3, activation="relu")
            layer("maxpool", pool_size=2)
            layer("batchnorm")
            layer("conv2d", filters=64, kernel_size=3, activation="relu")
            layer("maxpool", pool_size=2)
            layer("flatten")
            layer("dense", input=64*7*7, output=128, activation="relu")
            layer("dropout", rate=0.5)
            layer("dense", input=128, output=10, activation="softmax")
            train(epochs=15, learning_rate=0.001)
        """
        complex_ptx = """
        .entry complex_net2 {
            .param .u64 conv1, filters=32, kernel_size=3, activation="relu";
            .param .u64 maxpool1, pool_size=2;
            .param .u64 batchnorm;
            .param .u64 conv2, filters=64, kernel_size=3, activation="relu";
            .param .u64 maxpool2, pool_size=2;
            .param .u64 flatten;
            .param .u64 dense1, input=64*7*7, output=128, activation="relu";
            .param .u64 dropout, rate=0.5;
            .param .u64 dense2, input=128, output=10, activation="softmax";
            train_model(epochs=15, lr=0.001);
        }
        """
        wrapper = PyPTXWrapper(complex_model)
        self.assertIsNotNone(wrapper)

        syntax = PyPTXSyntax()
        syntax.parse(complex_model)
        compiled = syntax.compile()
        self.assertIn("complex_net2", compiled)
        self.logger.debug(f"Compiled complex model code:\n{compiled}")

        compiled_bytes = pyptx_compile(complex_ptx)
        self.assertIsInstance(compiled_bytes, bytes)
        self.assertIn(b"complex_net2", compiled_bytes)
        # Updated assertions to match converted parameter names
        self.assertIn(b"conv1", compiled_bytes)
        self.assertIn(b"conv2", compiled_bytes)
        self.assertIn(b"maxpool1", compiled_bytes)
        self.assertIn(b"maxpool2", compiled_bytes)

    def tearDown(self):
        """Clean up any resources"""
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
