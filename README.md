# Product Pulse: Advanced AI-Powered Feedback Analyzer

Product Pulse is a sophisticated Streamlit application that uses AI to analyze and summarize user feedback from various platforms like app stores, social media, and customer emails.

## Features

### Basic Features (simple_app.py)
- **Data Upload**: Upload feedback data in CSV or JSON format
- **Text Preprocessing**: Clean and prepare text for analysis
- **Smart Categorization**: Automatically classify feedback into pain points, feature requests, and positive feedback
- **Feedback Summarization**: Generate concise summaries for each feedback category
- **Interactive Visualization**: View feedback distribution and analysis results
- **CSV Export**: Generate and download comprehensive CSV reports

### Advanced Features (advanced_app.py)
- **Universal CSV Support**: Automatically detects delimiters and encodings for any CSV format
- **Special Format Support**: Built-in handler for Borderlands format (ID, Game, Sentiment, Text)
- **Multiple File Formats**: Support for CSV, JSON, Excel, and text files
- **Column Mapping**: Map non-standard column names to the expected format
- **Interactive Column Selection**: Select which column contains the feedback text
- **Sentiment Analysis**: Automatically detect positive, neutral, and negative sentiment
- **Topic Modeling**: Identify key themes in your feedback using LDA
- **Feedback Clustering**: Group similar feedback together using K-means
- **Word Clouds**: Visual representation of common terms by category and sentiment
- **Multiple Export Options**: Generate CSV, Excel, or PDF reports
- **Customizable Analysis**: Adjust parameters for sentiment threshold, topic modeling, and more

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd product-pulse
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install NLTK resources (required for advanced features):
   ```bash
   python install_nltk_resources.py
   ```

   If you encounter a `punkt_tab` error, run:
   ```bash
   python fix_punkt_tab.py
   ```

4. Run the application:

   **Option 1:** Using the provided scripts:

   For the basic version:
   ```bash
   ./run_basic.sh
   ```

   For the advanced version:
   ```bash
   ./run_advanced.sh
   ```

   **Option 2:** Manual execution:

   For the basic version:
   ```bash
   streamlit run simple_app.py
   ```

   For the advanced version:
   ```bash
   streamlit run advanced_app.py
   ```

## Data Format

### Standard Format
The application accepts CSV or JSON files with the following columns:
- `text`: The feedback text (required)
- `platform`: The source platform (optional, defaults to "Unknown")

Example:
```json
[
  {
    "text": "I love this app! It's so intuitive and helpful.",
    "platform": "App Store"
  },
  {
    "text": "The app keeps crashing when I try to upload photos.",
    "platform": "Google Play"
  }
]
```

### Borderlands Format
The advanced version also supports the Borderlands format CSV files with the following structure:
```
ID, Game, Sentiment, Text
```

Example:
```
2401, Borderlands, Positive, im getting on borderlands and i will murder you all
2402, Borderlands, Negative, this game keeps crashing every time I try to play online
```

### Other Formats
The advanced version can handle various CSV formats with:
- Different delimiters (comma, semicolon, tab, etc.)
- Different encodings (UTF-8, Latin-1, etc.)
- Files with or without headers
- Non-standard column names (with interactive mapping)

## How It Works

1. **Upload Data**: Upload your feedback data file
2. **Process Feedback**: Click the "Process Feedback" button to analyze the data
3. **Explore Results**: Navigate through the tabs to view raw data, categorized feedback, and summaries
4. **Generate Report**: Click "Generate CSV Report" to create and download a comprehensive report

## Technologies Used

### Basic Version
- **Streamlit**: For the web interface
- **Pandas**: For data manipulation
- **Matplotlib**: For data visualization
- **NumPy**: For numerical operations

### Advanced Version (Additional)
- **NLTK**: For natural language processing
- **TextBlob**: For sentiment analysis
- **scikit-learn**: For machine learning (topic modeling and clustering)
- **WordCloud**: For generating word clouds
- **Plotly**: For interactive visualizations
- **FPDF**: For PDF report generation
- **chardet**: For automatic encoding detection
- **openpyxl**: For Excel file handling

## Sample Data

A sample data file (`sample_data.json`) is included in the repository for testing purposes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
