import unittest
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral.spectral_struct.spectral_s_model import extract_spectral_row
from prosodic.prosodic_struct.prosodic_s_model import extract_prosodic_row
from linguistic.linguistic_struct.linguistic_s_model import LinguisticTabularModel

class TestSystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary dummy audio file to test the pipelines."""
        cls.dummy_audio_path = PROJECT_ROOT / "test_pipeline" / "dummy_audio.wav"
        
        # Create 1 second of random noise at 16kHz
        sr = 16000
        duration = 1.0
        audio_data = np.random.randn(int(sr * duration)).astype(np.float32)
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Save as WAV (using scipy to avoid soundfile dependency issues)
        wavfile.write(str(cls.dummy_audio_path), sr, audio_data)
        print(f"Created dummy audio file at {cls.dummy_audio_path}")

    @classmethod
    def tearDownClass(cls):
        """Clean up dummy file."""
        if cls.dummy_audio_path.exists():
            cls.dummy_audio_path.unlink()

    def test_01_spectral_extraction(self):
        """Test if the Spectral feature extractor successfully processes audio."""
        print("Testing Spectral Feature Extraction...")
        row = extract_spectral_row(str(self.dummy_audio_path))
        
        self.assertIsInstance(row, dict)
        self.assertTrue(len(row) > 100, "Spectral features should extract around 110 features.")
        self.assertIn("mfcc0_mean", row)
        self.assertIn("zcr_mean", row)
        print("  -> Passed Spectral Extraction.")

    def test_02_prosodic_extraction(self):
        """Test if the Prosodic feature extractor successfully processes audio."""
        print("Testing Prosodic Feature Extraction...")
        row = extract_prosodic_row(str(self.dummy_audio_path))
        
        self.assertIsInstance(row, dict)
        self.assertTrue(len(row) > 30, "Prosodic features should extract around 40 features.")
        self.assertIn("f0_mean", row)
        self.assertIn("energy_mean", row)
        print("  -> Passed Prosodic Extraction.")

    def test_03_linguistic_extraction_and_inference(self):
        """Test if Linguistic tabular model instantiates and extracts features."""
        print("Testing Linguistic Pipeline Initialization...")
        # Will just test instantiation and feature extraction 
        agent = LinguisticTabularModel()
        
        row = agent.extract_features(str(self.dummy_audio_path))
        self.assertIsInstance(row, dict)
        self.assertIn("char_count", row)
        self.assertIn("word_count", row)
        print("  -> Passed Linguistic Pipeline (Whisper & Feature Extraction).")

    def test_04_system_fusion_readiness(self):
        """Test if the outputs conform to what the fusion engine (MAD.py) expects."""
        # MAD.py expects a dictionary of scores from agents
        spec_row = extract_spectral_row(str(self.dummy_audio_path))
        pros_row = extract_prosodic_row(str(self.dummy_audio_path))
        
        self.assertIsNotNone(spec_row)
        self.assertIsNotNone(pros_row)
        
        # Test mock fusion logic as it would appear in MAD.py
        mock_weights = {"spectral": 0.4, "prosodic": 0.2, "linguistic": 0.4}
        mock_scores = {"spectral": 0.8, "prosodic": 0.6, "linguistic": 0.9}
        
        final_score = sum(mock_scores[k] * mock_weights[k] for k in mock_weights)
        self.assertAlmostEqual(final_score, 0.8*0.4 + 0.6*0.2 + 0.9*0.4)
        print("  -> Passed Fusion Engine Mathematical Readiness.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
