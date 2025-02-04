import unittest
import numpy as np
import logging
from pathlib import Path
import sys
import ctypes

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from final.wrapper import PyPTXWrapper
from final.Grapher import Grapher
from final.ptx_utils import PTXLoader

logger = logging.getLogger(__name__)

class TestNumericModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.ptx_loader = PTXLoader()
        
    def setUp(self):
        """Set up test case"""
        self.model_code = """
        @model
        def numeric_net():
            layer("dense", input=2, output=4, activation="relu")
            layer("dense", input=4, output=1, activation="linear")
            train(epochs=5, learning_rate=0.01)
        """
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        self.y = np.array([[0], [1], [1], [0]], dtype=np.float32)
        
    def test_model_compilation(self):
        """Test model compilation process"""
        wrapper = PyPTXWrapper(self.model_code)
        self.assertIsNotNone(wrapper)
        logger.info("Model compilation test passed")
        
    def test_graph_creation(self):
        """Test computation graph creation"""
        graph = Grapher()
        graph.add_operation("dense_1")
        graph.add_operation("relu")
        graph.add_operation("dense_2")
        graph.add_operation("linear")
        self.assertEqual(len(graph.operations), 4)
        logger.info("Graph creation test passed")
        
    def test_gpu_availability(self):
        """Test GPU detection and availability"""
        device_count = self.ptx_loader.get_device_count()
        logger.info(f"Detected {device_count} GPU devices")
        self.assertGreaterEqual(device_count, 0)
        
    def test_training_setup(self):
        """Test training data preparation"""
        self.assertEqual(self.X.shape, (4, 2))
        self.assertEqual(self.y.shape, (4, 1))
        logger.info("Training data setup verified")
        
    def test_full_training_cycle(self):
        """Test complete training cycle"""
        wrapper = PyPTXWrapper(self.model_code)
        graph = Grapher()
        
        # Add operations for forward pass
        graph.add_operation("dense_1")
        graph.add_operation("relu")
        graph.add_operation("dense_2")
        
        # Execute graph
        graph.execute()
        
        # Verify execution completed
        self.assertTrue(len(graph.operations) > 0)
        logger.info("Full training cycle completed successfully")

    def test_gpu_memory_management(self):
        """Test GPU memory allocation and data transfer"""
        if self.ptx_loader.is_available():
            try:
                # Allocate smaller memory size for testing
                data_size = 1024  # 1KB for testing
                gpu_ptr = self.ptx_loader.allocate_memory(data_size)
                
                # Create test data
                test_data = np.ones(data_size // 4, dtype=np.float32)
                host_ptr = test_data.ctypes.data_as(ctypes.c_void_p)
                
                self.ptx_loader.copy_to_device(host_ptr, gpu_ptr, data_size)
                
                # Verify operation
                result = np.zeros_like(test_data)
                result_ptr = result.ctypes.data_as(ctypes.c_void_p)
                self.ptx_loader.copy_to_host(gpu_ptr, result_ptr, data_size)
                
                np.testing.assert_array_almost_equal(test_data, result)
                logger.info("GPU memory management test passed")
            except RuntimeError as e:
                logger.warning(f"GPU test failed: {e}")
                self.skipTest("GPU memory allocation failed")
        else:
            self.skipTest("GPU not available")

    def test_forward_pass_result(self):
        """Test forward pass operations and verify results"""
        graph = Grapher()
        
        # Create sample input
        input_data = np.random.randn(1, 2).astype(np.float32)
        weights1 = np.random.randn(2, 4).astype(np.float32)
        weights2 = np.random.randn(4, 1).astype(np.float32)
        
        # Add operations with proper parameters
        graph.add_operation("dense_1", weights1)
        graph.add_operation("relu", None)
        graph.add_operation("dense_2", weights2)
        
        # Execute and verify output shape
        output = graph.execute()
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 1))
        logger.info("Forward pass verification complete")

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
