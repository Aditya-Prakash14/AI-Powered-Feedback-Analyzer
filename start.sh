#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Set default port if not provided
export PORT=${PORT:-8501}
echo "Using port: $PORT"

# Activate virtual environment if it exists
if [ -d "/opt/venv" ]; then
    echo "Activating virtual environment..."
    source /opt/venv/bin/activate
fi

# Download NLTK data
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create NLTK data directory if it doesn't exist
mkdir -p ~/nltk_data/tokenizers/

# Fix punkt_tab issue
echo "Fixing punkt_tab issue..."
if [ -f "fix_punkt_tab.py" ]; then
    python fix_punkt_tab.py
else
    echo "fix_punkt_tab.py not found, creating punkt_tab directory manually..."
    # Manual fix for punkt_tab issue
    if [ -d ~/nltk_data/tokenizers/punkt ]; then
        if [ ! -d ~/nltk_data/tokenizers/punkt_tab ]; then
            cp -r ~/nltk_data/tokenizers/punkt ~/nltk_data/tokenizers/punkt_tab
            echo "Created punkt_tab directory manually"
        fi
    fi
fi

# Start the Streamlit app
echo "Starting Streamlit app..."
streamlit run advanced_app.py --server.port=$PORT --server.address=0.0.0.0