#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install basic dependencies
echo "Installing basic dependencies..."
pip install streamlit pandas numpy matplotlib

# Run the basic app
echo "Starting the basic app..."
streamlit run simple_app.py
