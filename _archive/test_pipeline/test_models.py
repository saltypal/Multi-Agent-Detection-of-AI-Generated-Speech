import unittest
import torch
import numpy as np
from spectral.spectral_model import SpectralModel
from prosodic.prosodic_model import ProsodicModel

class TestModelArchitectures(unittest.TestCase):
    def test_spectral_forward(self):
        # (B, F, T) -> (B,)
        model = SpectralModel(n_features=100)
        dummy_input = torch.randn(2, 100, 300)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2,))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_prosodic_forward(self):
        # (B, F, T) -> (B,)
        model = ProsodicModel(n_features=5)
        dummy_input = torch.randn(2, 5, 300)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2,))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

if __name__ == "__main__":
    unittest.main()
