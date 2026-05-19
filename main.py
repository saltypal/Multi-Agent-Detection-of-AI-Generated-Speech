import sys
from pathlib import Path
from MAD import MultiAgentDetector

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: File {audio_path} not found.")
        sys.exit(1)
        
    # Initialize the system
    # This will load all models from 'trained_models' directory
    try:
        detector = MultiAgentDetector(models_root="trained_models")
        result = detector.detect(audio_path)
        
        print("\n" + "" + " "*2 + "DETECTION REPORT" + " "*2 + "")
        print("="*40)
        print(f"   FILE: {Path(audio_path).name}")
        print(f"   RESULT: {result['final_decision']}")
        print(f"   CONFIDENCE: {result['confidence']:.2%}")
        print("="*40)
        print("Agent Details:")
        for agent, score in result['agent_scores'].items():
            status = "FAKE" if score > 0.5 else "REAL"
            print(f" - {agent:<12}: {score:.4f} ({status})")
        print("="*40)
        
    except Exception as e:
        print(f"An error occurred during detection: {e}")

if __name__ == "__main__":
    main()
