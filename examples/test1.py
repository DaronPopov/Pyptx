import os
import sys
import logging
import unittest

# Configure logging with proper format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from final.pyptx_compiler import pyptx_compile
    from final.syntaxstructure import PyPTXSyntax
    from final.wrapper import PyPTXWrapper
    from final.Grapher import Grapher
except ImportError as e:
    logging.error("Could not import modules from pyptx. Check sys.path configuration.")
    raise e

class ComplexTest(unittest.TestCase):
    def test_compilation(self):
        """Test that pyptx_compile compiles code correctly and returns bytes."""
        sample_code = """
        .entry neural_net {
           .param .u64 dense, input=16, output=16, activation="relu";
           .param .u64 conv2d, filters=3, kernel_size=3;
           train_model(epochs=10, lr=0.01);
        }
        """
        compiled = pyptx_compile(sample_code)
        self.assertIsInstance(compiled, bytes)
        decoded = compiled.decode('utf-8')
        logging.debug(f"Decoded compiled code: {decoded}")
        self.assertIn("neural_net", decoded)
    
    def test_execution_graph(self):
        """Test Grapher functionality."""
        graph = Grapher()
        graph.add_operation("conv2d")
        graph.add_operation("relu")
        logging.debug(f"Graph operations: {graph.operations}")
        self.assertEqual(len(graph.operations), 2)
    
    def test_syntax_wrapper_integration(self):
        """Test basic integration between PyPTXSyntax and PyPTXWrapper."""
        syntax = PyPTXSyntax()
        wrapper = PyPTXWrapper()
        # In a real scenario, you'd expect methods for these classes.
        # For testing purposes, we at least check that instances exist.
        self.assertIsNotNone(syntax)
        self.assertIsNotNone(wrapper)
    
    def test_integration_flow(self):
        """Combine compilation and execution graph in a simulated flow."""
        sample_code = """
        .entry neural_net {
           .param .u64 dense, input=16, output=16, activation="relu";
           .param .u64 conv2d, filters=3, kernel_size=3;
           train_model(epochs=5, lr=0.005);
        }
        """
        compiled = pyptx_compile(sample_code)
        decoded = compiled.decode('utf-8')
        self.assertIn("neural_net", decoded)
        
        graph = Grapher()
        graph.add_operation("train_model")
        # Execute to simulate a full run; logging output serves as our trace.
        graph.execute()
        self.assertEqual(len(graph.operations), 1)

if __name__ == '__main__':
    unittest.main()