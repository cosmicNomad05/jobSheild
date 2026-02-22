#!/bin/bash
set -e  # stop on any error

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Downloading model from Hugging Face Hub ==="
python -c "
from huggingface_hub import hf_hub_download
import os

REPO_ID = 'vkxaHere/jobshield-model'
os.makedirs('model', exist_ok=True)

print('Downloading model.pkl ...')
hf_hub_download(repo_id=REPO_ID, filename='model.pkl', local_dir='model')

print('Downloading vectorizer.pkl ...')
hf_hub_download(repo_id=REPO_ID, filename='vectorizer.pkl', local_dir='model')

print('Model files ready.')
"

echo "=== Build complete ==="
