#!/bin/bash

# Download NLTK data
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Fix punkt_tab issue
echo "Fixing punkt_tab issue..."
python fix_punkt_tab.py

# Start the Streamlit app
echo "Starting Streamlit app..."
streamlit run advanced_app.py --server.port=$PORT --server.address=0.0.0.0