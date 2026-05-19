import os
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, render_template

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MAD import MultiAgentDetector

app = Flask(__name__)
detector = None

UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

def get_detector():
    global detector
    if detector is None:
        # Load detector pointing to local trained_models directory
        detector = MultiAgentDetector(models_root=PROJECT_ROOT / "trained_models")
    return detector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded.'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    filepath = UPLOAD_FOLDER / file.filename
    try:
        file.save(str(filepath))
        
        # Initialize/Warm up detector
        det = get_detector()
        
        # Execute multi-agent inference
        result = det.detect(str(filepath))
        
        # Clean up file after inference
        if filepath.exists():
            os.remove(str(filepath))
            
        return jsonify(result)
        
    except Exception as e:
        if filepath.exists():
            os.remove(str(filepath))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[*] Warming up Multi-Agent models...")
    try:
        get_detector()
    except Exception as e:
        print(f"[!] Warning during model warmup: {e}")
        
    print("[*] Starting Local Web Dashboard on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
