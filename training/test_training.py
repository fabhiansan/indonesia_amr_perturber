"""
Test script for verifying the AMR entailment training process.
"""
import os
import json
import unittest
import tempfile
import subprocess
from pathlib import Path

class TestTrainingProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(__file__).parent
        cls.dummy_data_path = cls.test_dir / "dummy_dataset.json"
        cls.output_dir = tempfile.mkdtemp(prefix="amr_test_output_")
        
    def test_data_loading(self):
        """Test that the dummy dataset can be loaded"""
        with open(self.dummy_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Check required fields
        for example in data:
            self.assertIn("source_text", example)
            self.assertIn("amr", example)
            self.assertIn("score", example)
    
    def test_training_process(self):
        """Test that training runs without errors"""
        cmd = [
            "python", str(self.test_dir / "train.py"),
            "--data_path", str(self.dummy_data_path),
            "--output_dir", self.output_dir,
            "--model_name_or_path", "indobenchmark/indobert-base-p1",
            "--batch_size", "2",
            "--num_train_epochs", "1",
            "--no_cuda"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check training completed successfully
        self.assertEqual(result.returncode, 0, 
                         f"Training failed with error:\n{result.stderr}")
        
        # Check output files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test_results.json")),
                        "Test results file not found")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test directory"""
        # Remove temporary output directory
        import shutil
        shutil.rmtree(cls.output_dir)

if __name__ == "__main__":
    unittest.main()