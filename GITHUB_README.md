# Product Pulse: AI-Powered Feedback Analyzer

![Product Pulse Banner](https://via.placeholder.com/1200x300/4a90e2/ffffff?text=Product+Pulse:+AI-Powered+Feedback+Analyzer)

Product Pulse is a sophisticated application that uses AI to analyze and summarize user feedback from various platforms like app stores, social media, and customer emails.

## 🌟 Features

### Basic Features
- **Data Upload**: Upload feedback data in CSV or JSON format
- **Text Preprocessing**: Clean and prepare text for analysis
- **Smart Categorization**: Automatically classify feedback into pain points, feature requests, and positive feedback
- **Feedback Summarization**: Generate concise summaries for each feedback category
- **Interactive Visualization**: View feedback distribution and analysis results
- **CSV Export**: Generate and download comprehensive CSV reports

### Advanced Features
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

## 📋 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/product-pulse.git
   cd product-pulse
   ```

2. Install the required dependencies:
   ```bash
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
   
   For the basic version:
   ```bash
   streamlit run simple_app.py
   ```
   
   For the advanced version:
   ```bash
   streamlit run advanced_app.py
   ```

   Or use the provided scripts:
   ```bash
   ./run_basic.sh    # For basic version
   ./run_advanced.sh # For advanced version
   ```

## 📊 Screenshots

![Dashboard](https://via.placeholder.com/800x450/4a90e2/ffffff?text=Dashboard)
![Sentiment Analysis](https://via.placeholder.com/800x450/4a90e2/ffffff?text=Sentiment+Analysis)
![Topic Modeling](https://via.placeholder.com/800x450/4a90e2/ffffff?text=Topic+Modeling)

## 🔍 Data Format

### Standard Format
The application accepts CSV or JSON files with the following columns:
- `text`: The feedback text (required)
- `platform`: The source platform (optional, defaults to "Unknown")

### Borderlands Format
The advanced version also supports the Borderlands format CSV files with the following structure:
```
ID, Game, Sentiment, Text
```

### Other Formats
The advanced version can handle various CSV formats with:
- Different delimiters (comma, semicolon, tab, etc.)
- Different encodings (UTF-8, Latin-1, etc.)
- Files with or without headers
- Non-standard column names (with interactive mapping)

## 🛠️ Technologies Used

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

If you have any questions or feedback, please open an issue on GitHub.
