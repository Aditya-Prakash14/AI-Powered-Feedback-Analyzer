#!/bin/bash

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Fix punkt_tab issue
python fix_punkt_tab.py

# Start the Streamlit app
streamlit run advanced_app.py