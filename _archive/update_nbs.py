import json
from pathlib import Path

def update_notebook(path_str):
    path = Path(path_str)
    if not path.exists(): return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    new_source = [
        "# Download dataset using curl\n",
        "DATA_DIR = Path('/content/data/asvspoof5')\n",
        "ZIP_PATH = Path('/content/data/asvspoof5-flac.zip')\n",
        "\n",
        "if DATA_DIR.exists() and any(DATA_DIR.iterdir()):\n",
        "    print(f'[✔] Dataset already exists at {DATA_DIR}')\n",
        "else:\n",
        "    DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
        "    print(f'[↓] Downloading dataset using curl...')\n",
        "    subprocess.run([\n",
        "        'curl', '-L', '-o', str(ZIP_PATH), \n",
        "        'https://www.kaggle.com/api/v1/datasets/download/aniket202411001/asvspoof5-flac'\n",
        "    ], check=True)\n",
        "    print(f'[↓] Unzipping dataset...')\n",
        "    subprocess.run(['unzip', '-q', str(ZIP_PATH), '-d', str(DATA_DIR)], check=True)\n",
        "    ZIP_PATH.unlink()  # Remove zip file to save space\n",
        "    print('[✔] Download and extraction complete.')\n"
    ]
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('KAGGLE_DATASET =' in line for line in cell['source']):
            cell['source'] = new_source
            break
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

update_notebook(r'd:\Bunker\BaseCamp\Multi-Agent-Detection-of-AI-Generated-Speech\spectral\spectral_struct\extract_features.ipynb')
update_notebook(r'd:\Bunker\BaseCamp\Multi-Agent-Detection-of-AI-Generated-Speech\prosodic\prosodic_struct\extract_features.ipynb')
update_notebook(r'd:\Bunker\BaseCamp\Multi-Agent-Detection-of-AI-Generated-Speech\linguistic\linguistic_struct\extract_features.ipynb')
