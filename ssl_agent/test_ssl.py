import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import the ssl module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ssl_agent.ssl_model import SSLAgent

def generate_dummy_audio(filename="dummy_test.wav"):
    """Generates a 1-second 16kHz sine wave audio file for testing if none exists."""
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    sf.write(filename, audio, sr)
    return filename

def main():
    print("=== Testing SSL Agent Wrapper ===")
    
    # Initialize the Agent
    print("\n1. Initializing SSLAgent...")
    try:
        agent = SSLAgent()
        print("✅ Successfully loaded WavLM backbone and feature extractor!")
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        return

    # Create dummy audio
    audio_path = "dummy_test.wav"
    if not os.path.exists(audio_path):
        generate_dummy_audio(audio_path)
    
    # Test inference
    print(f"\n2. Testing Inference on {audio_path}...")
    try:
        probability = agent.predict(audio_path)
        print(f"✅ Success! Spoof Probability: {probability:.4f} ({probability * 100:.2f}%)")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
    finally:
        # Cleanup
        if os.path.exists("dummy_test.wav"):
            os.remove("dummy_test.wav")

if __name__ == "__main__":
    main()
