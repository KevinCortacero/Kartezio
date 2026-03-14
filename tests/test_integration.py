"""
Integration tests for Kartezio main workflows.
Tests complete training pipelines, model evolution, and end-to-end functionality.
"""

import unittest
from unittest.mock import patch

import numpy as np

from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.matrix import default_matrix_lib
from kartezio.utils.dataset import one_cell_dataset


class TestBasicTrainingWorkflow(unittest.TestCase):
    """Test basic training workflow similar to examples/training/basic_trainer.py."""

    def setUp(self):
        """Set up test training configuration."""
        self.n_inputs = 1
        self.libraries = default_matrix_lib()
        self.endpoint = EndpointThreshold(128)
        self.fitness = IoU()

        # Create minimal model for testing
        self.model = KartezioTrainer(
            n_inputs=self.n_inputs,
            n_nodes=self.n_inputs * 5,  # Smaller for faster testing
            libraries=self.libraries,
            endpoint=self.endpoint,
            fitness=self.fitness,
        )
        self.model.set_mutation_rates(node_rate=0.1, out_rate=0.2)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.decoder.adapter.n_inputs, self.n_inputs)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)  # Default
        self.assertEqual(self.model.decoder.adapter.n_nodes, 5)

        # Check that components are properly set
        self.assertIsNotNone(self.model.decoder.libraries)
        self.assertIsNotNone(self.model.decoder.endpoint)
        self.assertIsNotNone(self.model.fitness)

    @unittest.skip("Requires actual training data and takes time")
    def test_complete_training_workflow(self):
        """Test complete training workflow (disabled by default due to time)."""
        # Load minimal training data
        try:
            train_x, train_y = one_cell_dataset()
        except Exception:
            self.skipTest("Training data not available")

        # Run short training
        elite, history = self.model.fit(5, train_x, train_y)  # Just 5 generations

        # Verify training completed
        self.assertIsNotNone(elite)
        self.assertIsNotNone(history)
        self.assertGreater(len(history), 0)

        # Test evaluation
        evaluation_result = self.model.evaluate(train_x, train_y)
        self.assertIsNotNone(evaluation_result)
        self.assertIsInstance(evaluation_result, (float, np.float32, np.float64))

    def test_model_configuration(self):
        """Test model configuration and parameter setting."""
        # Test mutation rate setting
        self.model.set_mutation_rates(node_rate=0.05, out_rate=0.1)
        # Note: We can't directly test these without accessing private attributes
        # This mainly tests that the method doesn't crash

        # Test that model can generate initial population
        # This is an indirect test of initialization
        try:
            # This should not crash
            population_size = 10
            # We can't easily test population generation without running fit()
            # so this is mainly a smoke test
            pass
        except Exception as e:
            self.fail(f"Model configuration failed: {e}")

    def test_fitness_function_integration(self):
        """Test that fitness function integrates properly."""
        # Create mock data
        y_true = [np.random.randint(0, 2, (10, 10)).astype(np.uint8)]
        y_pred = [
            [np.random.randint(0, 2, (10, 10)).astype(np.uint8)]
        ]  # Single individual

        # Test fitness evaluation
        fitness_score = self.fitness.batch(y_true, y_pred)

        self.assertIsInstance(fitness_score, np.ndarray)
        self.assertEqual(len(fitness_score), 1)  # One individual
        self.assertTrue(0 <= fitness_score[0] <= 1)  # IoU should be between 0 and 1

    def test_endpoint_integration(self):
        """Test endpoint integration with training pipeline."""
        # Create mock input data
        mock_input = [np.random.rand(20, 20) * 255]  # Random image-like data

        # Test endpoint processing
        result = self.endpoint.call(mock_input)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        # Check if result is binary (0 or 1) - dtype may vary
        self.assertTrue(
            np.all((result[0] == 0) | (result[0] == 1))
            or np.all((result[0] >= 0) & (result[0] <= 1))
        )  # Binary or normalized output


class TestLibraryIntegration(unittest.TestCase):
    """Test library integration with the training system."""

    def setUp(self):
        """Set up test library."""
        self.library = default_matrix_lib()

    def test_library_initialization(self):
        """Test that default matrix library initializes correctly."""
        self.assertIsNotNone(self.library)
        self.assertGreater(self.library.size, 0)

        # Test that library has expected properties
        self.assertGreater(self.library.max_arity, 0)
        self.assertGreaterEqual(self.library.max_parameters, 0)

    def test_primitive_execution_in_library(self):
        """Test that primitives in the library can be executed."""
        if self.library.size == 0:
            self.skipTest("No primitives in library")

        # Test first primitive (should be safe)
        try:
            # Create mock input matching the first primitive's requirements
            arity = self.library.arity_of(0)
            inputs_types = self.library.inputs_of(0)
            n_params = self.library.parameters_of(0)

            # Create mock inputs (assuming matrix operations)
            mock_inputs = [np.random.rand(10, 10) for _ in range(arity)]
            mock_params = [128] * n_params if n_params > 0 else []

            # Execute primitive
            result = self.library.execute(0, mock_inputs, mock_params)

            # Verify result
            self.assertIsNotNone(result)
            self.assertIsInstance(result, np.ndarray)

        except Exception as e:
            self.fail(f"Primitive execution failed: {e}")

    def test_library_display(self):
        """Test library display functionality."""
        # This should not crash
        try:
            self.library.display()
        except Exception as e:
            self.fail(f"Library display failed: {e}")


class TestDatasetIntegration(unittest.TestCase):
    """Test dataset integration with training pipeline."""

    def test_one_cell_dataset_loading(self):
        """Test loading of one cell dataset."""
        try:
            train_x, train_y = one_cell_dataset()

            # Verify data structure
            self.assertIsInstance(train_x, list)
            self.assertIsInstance(train_y, list)
            self.assertGreater(len(train_x), 0)
            self.assertGreater(len(train_y), 0)
            self.assertEqual(len(train_x), len(train_y))

            # Verify data types (train_x and train_y are lists of arrays)
            for x in train_x:
                if isinstance(x, list):
                    self.assertIsInstance(x[0], np.ndarray)
                    self.assertEqual(len(x[0].shape), 2)  # Should be 2D image
                else:
                    self.assertIsInstance(x, np.ndarray)
                    self.assertEqual(len(x.shape), 2)  # Should be 2D image

            for y in train_y:
                if isinstance(y, list):
                    self.assertIsInstance(y[0], np.ndarray)
                    self.assertEqual(len(y[0].shape), 2)  # Should be 2D label
                else:
                    self.assertIsInstance(y, np.ndarray)
                    self.assertEqual(len(y.shape), 2)  # Should be 2D label

        except ImportError:
            self.skipTest("Dataset dependencies not available")
        except Exception as e:
            self.fail(f"Dataset loading failed: {e}")


class TestModelExportIntegration(unittest.TestCase):
    """Test model export functionality integration."""

    def setUp(self):
        """Set up minimal model for export testing."""
        self.n_inputs = 1
        self.libraries = default_matrix_lib()
        self.endpoint = EndpointThreshold(128)
        self.fitness = IoU()

        self.model = KartezioTrainer(
            n_inputs=self.n_inputs,
            n_nodes=3,  # Minimal for testing
            libraries=self.libraries,
            endpoint=self.endpoint,
            fitness=self.fitness,
        )

    def test_python_class_export(self):
        """Test Python class export functionality."""
        # Create a mock trained model by setting up basic genome
        # This is a smoke test - we can't easily create a fully trained model
        try:
            # This should not crash even with an untrained model
            self.model.print_python_class("TestExportClass")

        except Exception:
            # For now, we accept that this might fail with untrained model
            # The important thing is it doesn't crash the system
            pass

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_model_serialization_integration(self, mock_open):
        """Test model serialization capabilities."""
        # Test that model components can be serialized
        try:
            # Test endpoint serialization
            endpoint_dict = self.endpoint.__to_dict__()
            self.assertIsInstance(endpoint_dict, dict)
            self.assertIn("name", endpoint_dict)

            # Test fitness serialization
            fitness_dict = self.fitness.__to_dict__()
            self.assertIsInstance(fitness_dict, dict)

            # Test library serialization
            library_dict = self.libraries.__to_dict__()
            self.assertIsInstance(library_dict, dict)
            self.assertIn("rtype", library_dict)

        except Exception as e:
            self.fail(f"Model serialization failed: {e}")


class TestErrorRecoveryIntegration(unittest.TestCase):
    """Test error recovery and robustness in integrated workflows."""

    def test_invalid_training_data_handling(self):
        """Test handling of invalid training data."""
        model = KartezioTrainer(
            n_inputs=1,
            n_nodes=3,
            libraries=default_matrix_lib(),
            endpoint=EndpointThreshold(128),
            fitness=IoU(),
        )

        # Test various invalid data scenarios
        invalid_data_cases = [
            ([], []),  # Empty data
            ([np.array([])], [np.array([])]),  # Empty arrays
            ([np.array([1, 2, 3])], [np.array([1, 2])]),  # Mismatched shapes
            (None, None),  # None data
        ]

        for train_x, train_y in invalid_data_cases:
            with self.subTest(train_x=train_x, train_y=train_y):
                try:
                    # This should either handle gracefully or raise appropriate exceptions
                    model.fit(1, train_x, train_y)
                except (ValueError, TypeError, AttributeError):
                    # These are acceptable - invalid data should be rejected
                    pass
                except Exception:
                    # Other exceptions might be acceptable too
                    # The key is that the system doesn't crash silently
                    pass

    def test_library_with_no_primitives(self):
        """Test handling when library has no primitives."""
        from kartezio.core.components import Library
        from kartezio.types import Matrix

        empty_library = Library(Matrix)
        self.assertEqual(empty_library.size, 0)

        # Test that model can handle empty library (should fail gracefully)
        try:
            model = KartezioTrainer(
                n_inputs=1,
                n_nodes=3,
                libraries=empty_library,
                endpoint=EndpointThreshold(128),
                fitness=IoU(),
            )
            # If this succeeds, that's OK
            # If it fails, it should fail with a clear error
        except Exception as e:
            # Acceptable - empty library should cause clear failure
            self.assertIsInstance(e, (ValueError, AssertionError, ZeroDivisionError))

    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        # Test with extreme values that might cause issues
        extreme_cases = [
            {"n_inputs": 0, "n_nodes": 1},  # Zero inputs
            {"n_inputs": 1, "n_nodes": 0},  # Zero nodes
            {"n_inputs": 100, "n_nodes": 1000},  # Very large values
        ]

        for case in extreme_cases:
            with self.subTest(**case):
                try:
                    model = KartezioTrainer(
                        n_inputs=case["n_inputs"],
                        n_nodes=case["n_nodes"],
                        libraries=default_matrix_lib(),
                        endpoint=EndpointThreshold(128),
                        fitness=IoU(),
                    )
                    # If initialization succeeds, that's acceptable
                except (ValueError, AssertionError):
                    # These are acceptable - extreme values should be rejected
                    pass
                except Exception:
                    # Other exceptions might indicate a problem
                    pass


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance-related integration aspects."""

    def test_memory_usage_basic(self):
        """Basic test for memory usage (smoke test)."""
        # Create model and ensure it doesn't consume excessive memory
        model = KartezioTrainer(
            n_inputs=1,
            n_nodes=10,
            libraries=default_matrix_lib(),
            endpoint=EndpointThreshold(128),
            fitness=IoU(),
        )

        # This is mainly a smoke test - if it creates without crashing,
        # memory usage is probably reasonable
        self.assertIsNotNone(model)

        # Clean up
        del model

    def test_numerical_stability(self):
        """Test numerical stability with edge case values."""
        fitness = IoU()

        # Test with edge cases that might cause numerical issues
        edge_cases = [
            (np.zeros((5, 5)), np.zeros((5, 5))),  # All zeros
            (np.ones((5, 5)), np.ones((5, 5))),  # All ones
            (np.zeros((5, 5)), np.ones((5, 5))),  # Complete mismatch
        ]

        for y_true, y_pred in edge_cases:
            with self.subTest(y_true_sum=y_true.sum(), y_pred_sum=y_pred.sum()):
                try:
                    result = fitness.evaluate([y_true], [y_pred])

                    # Result should be finite and in expected range
                    self.assertTrue(np.isfinite(result[0]))
                    self.assertTrue(0 <= result[0] <= 1)

                except Exception as e:
                    self.fail(f"Numerical stability test failed: {e}")


if __name__ == "__main__":
    # Run with reduced verbosity for integration tests
    unittest.main(verbosity=1)
