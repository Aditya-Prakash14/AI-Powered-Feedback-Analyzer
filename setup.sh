#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data for the advanced version
echo "Downloading NLTK data for advanced version..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo "Setup complete!"
echo "Run 'streamlit run simple_app.py' for the basic version."
echo "Run 'streamlit run advanced_app.py' for the advanced version with more features."
