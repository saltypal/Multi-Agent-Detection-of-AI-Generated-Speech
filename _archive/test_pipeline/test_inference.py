import unittest
import os
import shutil
from pathlib import Path
from MAD import MADSystem

class TestMADInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create dummy results to test weight calculation
        for agent in ["spectral", "prosodic", "linguistic"]:
            res_dir = Path(agent) / "results"
            res_dir.mkdir(parents=True, exist_ok=True)
            with open(res_dir / "results.txt", "w") as f:
                # Give spectral the best EER
                eer = 0.05 if agent == "spectral" else 0.2
                f.write(f"EER: {eer}\n")

    def test_weight_calculation(self):
        mad = MADSystem()
        weights = mad.weights
        # Spectral should have highest weight because of lowest EER
        self.assertGreater(weights["spectral"], weights["prosodic"])
        self.assertGreater(weights["spectral"], weights["linguistic"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    @classmethod
    def tearDownClass(cls):
        # Cleanup dummy results if needed
        pass

if __name__ == "__main__":
    unittest.main()
