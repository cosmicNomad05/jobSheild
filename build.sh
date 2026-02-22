#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading dataset..."
# If you've uploaded your CSV somewhere public (Google Drive, etc.), download it here.
# Example with gdown (Google Drive):
pip install gdown
gdown "1uEeG7EreNuv3YI5O-XOt8DITOfLAg9qT" -O data/fake_job_postings.csv

echo "Training model..."
python train.py --data data/fake_job_postings.csv

echo "Build complete."