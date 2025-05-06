#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install NLTK resources
echo "Installing NLTK resources..."
python install_nltk_resources.py

# Fix punkt_tab issue
echo "Fixing punkt_tab issue..."
python fix_punkt_tab.py

# Run the advanced app
echo "Starting the advanced app..."
streamlit run advanced_app.py
