
# Installation Requirements:
1. System Dependecies 
```
sudo apt install git-lfs
```
2. Python Packages 
```
git lfs pull
```
3. Hugging Face Authentication
```
hf auth login
```

4. Create Required Directiores
```
mkdir -p retrieval/output
```

# Run the app

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Launch the Gradio app:

```bash
python3 app.py
```

Open http://localhost:7860 in your browser to access the interface.