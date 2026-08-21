import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fine_tuning.dataset import dataset_builder
from src.fine_tuning.train_qlora import qlora_trainer
from src.fine_tuning.export import model_exporter

class TestFineTuningPipeline(unittest.TestCase):
    def test_dataset_formatting(self):
        samples = dataset_builder.get_sample_compliance_data()
        self.assertGreater(len(samples), 0)
        
        formatted = dataset_builder.format_llama3(samples[0])
        self.assertIn("<|start_header_id|>system<|end_header_id|>", formatted)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", formatted)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", formatted)

    def test_trainer_config_and_parameters(self):
        config = qlora_trainer.get_config_summary()
        self.assertIn("trainable_percentage", config)
        self.assertLess(config["trainable_percentage"], 2.0)
        self.assertEqual(config["quantization"], "4-bit (NF4 with double quantization)")

    def test_simulated_training_loop(self):
        result = qlora_trainer.train_simulation(epochs=2)
        self.assertEqual(result["status"], "completed")
        self.assertLess(result["final_loss"], result["initial_loss"])
        self.assertEqual(len(result["loss_history"]), result["total_steps"])

    def test_export_modelfile_generation(self):
        modelfile = model_exporter.generate_modelfile("meta-llama/Llama-3.2-1B-Instruct", "./adapters/compliance_v1")
        self.assertIn("FROM meta-llama/Llama-3.2-1B-Instruct", modelfile)
        self.assertIn("ADAPTER ./adapters/compliance_v1", modelfile)

if __name__ == "__main__":
    unittest.main()
