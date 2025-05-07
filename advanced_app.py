# Import streamlit
import streamlit as st

# Import other libraries
import pandas as pd
import numpy as np
import re
import json
import os
import io
from io import BytesIO
import base64
import chardet
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from fpdf import FPDF
import datetime
import time
import tempfile

# Import the special handler for Borderlands format
try:
    from borderlands_handler import process_borderlands_format, is_borderlands_format
except ImportError:
    # Define fallback functions if the module is not available
    def process_borderlands_format(file_path):
        return pd.read_csv(file_path, header=None, names=['id', 'platform', 'sentiment', 'text'])

    def is_borderlands_format(file_content):
        return False

# Initialize NLTK resources
nltk_resources_available = True

# Create a container for NLTK initialization messages
nltk_init_container = st.empty()

try:
    import nltk

    # Try to import NLTK components
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        # Test if resources are available
        stopwords.words('english')
        word_tokenize("Test sentence")
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("testing")

        # If we get here, all resources are available
        with nltk_init_container.container():
            st.success("✅ NLTK resources are available and working properly.")
            time.sleep(1)  # Show success message briefly

    except (ImportError, LookupError) as e:
        # Resources not available, download them
        with nltk_init_container.container():
            st.info("Downloading required NLTK resources...")

            # Download required resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            # Try importing again
            try:
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                from nltk.stem import WordNetLemmatizer

                # Test if resources are now available
                stopwords.words('english')
                word_tokenize("Test sentence")
                lemmatizer = WordNetLemmatizer()
                lemmatizer.lemmatize("testing")

                st.success("✅ NLTK resources downloaded and initialized successfully.")
                time.sleep(1)  # Show success message briefly
            except Exception as inner_e:
                st.warning(f"⚠️ Some NLTK features may be limited: {str(inner_e)}")
                nltk_resources_available = False

except Exception as e:
    with nltk_init_container.container():
        st.warning(f"⚠️ NLTK initialization failed: {str(e)}")
    nltk_resources_available = False

# Clear the container after initialization
nltk_init_container.empty()

# Initialize session state variables if they don't exist
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'topics' not in st.session_state:
        st.session_state.topics = {}
    if 'config' not in st.session_state:
        st.session_state.config = {
            'sentiment_threshold': 0.1,
            'min_topic_words': 5,
            'num_topics': 3,
            'num_clusters': 4,
            'min_word_length': 3
        }

# Initialize session state
initialize_session_state()

# Function to detect CSV delimiter
def detect_delimiter(file_content):
    # Common delimiters to check
    delimiters = [',', ';', '\t', '|']
    counts = {delimiter: file_content.count(delimiter) for delimiter in delimiters}
    return max(counts, key=counts.get)

# Function to detect file encoding
def detect_encoding(file_content):
    result = chardet.detect(file_content)
    return result['encoding']

# Function to preprocess text
def preprocess_text(text, remove_stopwords=True):
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove hashtags and mentions
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    # Advanced processing with NLTK if available
    if remove_stopwords and nltk_resources_available:
        try:
            # Import NLTK components inside the function to avoid issues
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer

            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words and len(word) > st.session_state.config['min_word_length']]

            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

            text = ' '.join(tokens)
        except Exception as e:
            # Fallback to simple word filtering if NLTK fails
            # Simple fallback without NLTK
            words = text.split()
            # Filter out very short words
            words = [word for word in words if len(word) > st.session_state.config['min_word_length']]
            text = ' '.join(words)
    else:
        # Simple processing without NLTK
        words = text.split()
        # Filter out very short words
        words = [word for word in words if len(word) > st.session_state.config['min_word_length']]
        text = ' '.join(words)

    return text

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        analysis = TextBlob(text)

        # Determine sentiment category
        if analysis.sentiment.polarity > st.session_state.config['sentiment_threshold']:
            return 'positive'
        elif analysis.sentiment.polarity < -st.session_state.config['sentiment_threshold']:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        # Simple rule-based fallback for sentiment analysis
        st.warning(f"TextBlob sentiment analysis failed: {str(e)}. Using simple keyword-based analysis.")

        text = text.lower()
        # Define positive and negative keywords
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic', 'wonderful', 'happy', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor', 'disappointing', 'disappointed', 'useless', 'problem', 'issue', 'bug', 'crash']

        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        # Determine sentiment based on counts
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

# Advanced categorization using sentiment and keywords
def categorize_feedback(text, sentiment):
    text = text.lower()

    # Define keywords for each category
    pain_points = ['crash', 'bug', 'error', 'slow', 'issue', 'problem', 'fix', 'broken', 'annoying', 'difficult', 'bad', 'terrible', 'awful', 'horrible', 'poor']
    feature_requests = ['add', 'feature', 'would be', 'could you', 'please', 'implement', 'integration', 'support', 'option', 'wish', 'hope', 'want', 'need', 'should have', 'missing']
    positive_feedback = ['love', 'great', 'awesome', 'amazing', 'good', 'excellent', 'best', 'fantastic', 'helpful', 'intuitive', 'easy', 'simple', 'nice', 'perfect', 'wonderful']

    # Count matches for each category
    pain_count = sum(1 for word in pain_points if word in text)
    feature_count = sum(1 for word in feature_requests if word in text)
    positive_count = sum(1 for word in positive_feedback if word in text)

    # Weight by sentiment
    if sentiment == 'negative':
        pain_count *= 1.5
    elif sentiment == 'positive':
        positive_count *= 1.5

    # Determine the category with the most matches
    counts = [pain_count, feature_count, positive_count]
    categories = ['pain point', 'feature request', 'positive feedback']

    if max(counts) == 0:
        return 'uncategorized'

    return categories[counts.index(max(counts))]

# Function to extract topics using LDA
def extract_topics(texts, num_topics=3, num_words=5):
    if not texts or len(texts) < 3:
        return [["Not enough data for topic modeling"]]

    try:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Transform the texts to TF-IDF features
        X = vectorizer.fit_transform(texts)

        # Create and fit the LDA model
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Extract top words for each topic
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)

        return topics
    except Exception as e:
        st.warning(f"Topic modeling failed: {str(e)}. Using simple word frequency analysis instead.")
        # Fallback to simple word frequency analysis
        try:
            # Count word frequencies across all texts
            word_freq = {}
            for text in texts:
                if isinstance(text, str):
                    words = text.split()
                    for word in words:
                        if len(word) > st.session_state.config['min_word_length']:
                            word_freq[word] = word_freq.get(word, 0) + 1

            # Sort words by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

            # Create simple topics based on word frequency
            topics = []
            words_per_topic = min(num_words, len(sorted_words) // num_topics)

            for i in range(num_topics):
                start_idx = i * words_per_topic
                end_idx = start_idx + words_per_topic
                if start_idx < len(sorted_words):
                    topic_words = [w[0] for w in sorted_words[start_idx:end_idx]]
                    topics.append(topic_words)
                else:
                    topics.append(["Not enough data"])

            return topics
        except:
            return [["Simple topic analysis failed"]]

# Function to cluster feedback
def cluster_feedback(texts, n_clusters=4):
    if not texts or len(texts) < n_clusters:
        return np.zeros(len(texts) if texts else 0)

    try:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Transform the texts to TF-IDF features
        X = vectorizer.fit_transform(texts)

        # Create and fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        return clusters
    except Exception as e:
        st.warning(f"Clustering failed: {str(e)}. Using simple clustering instead.")
        # Fallback to simple clustering based on text length
        try:
            # Group texts by length as a very simple clustering method
            text_lengths = [len(text) if isinstance(text, str) else 0 for text in texts]

            # Determine length thresholds for clustering
            if text_lengths:
                min_len = min(text_lengths)
                max_len = max(text_lengths)
                range_len = max_len - min_len

                # Create clusters based on text length
                clusters = []
                for length in text_lengths:
                    if range_len == 0:
                        # If all texts are the same length, assign to first cluster
                        cluster = 0
                    else:
                        # Normalize length to cluster index
                        normalized = (length - min_len) / range_len
                        cluster = int(normalized * (n_clusters - 1))
                    clusters.append(cluster)

                return np.array(clusters)
            else:
                return np.zeros(len(texts))
        except:
            return np.zeros(len(texts))

# Advanced summarization function
def summarize_feedback(texts, category, sentiment_counts=None):
    if not texts:
        return f"No {category} feedback identified."

    if len(texts) == 1:
        return texts[0]

    # Extract topics
    topics = extract_topics(texts, st.session_state.config['num_topics'], st.session_state.config['min_topic_words'])

    # Store topics for later use
    st.session_state.topics[category] = topics

    # Count word frequency
    word_freq = {}
    for text in texts:
        words = text.lower().split()
        for word in words:
            if len(word) > st.session_state.config['min_word_length']:
                word_freq[word] = word_freq.get(word, 0) + 1

    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = [w[0] for w in sorted_words[:10] if w[1] > 1]

    # Generate a summary based on the category and sentiment
    summary = f"Analysis of {len(texts)} {category} items:\n\n"

    # Add sentiment distribution if available
    if sentiment_counts:
        total = sum(sentiment_counts.values())
        sentiment_percentages = {k: (v/total*100) for k, v in sentiment_counts.items()}
        summary += f"Sentiment: {sentiment_percentages.get('positive', 0):.1f}% Positive, "
        summary += f"{sentiment_percentages.get('neutral', 0):.1f}% Neutral, "
        summary += f"{sentiment_percentages.get('negative', 0):.1f}% Negative\n\n"

    # Add key topics
    summary += "Key topics identified:\n"
    for i, topic_words in enumerate(topics):
        summary += f"- Topic {i+1}: {', '.join(topic_words)}\n"

    # Add category-specific insights
    if category == 'pain point':
        summary += f"\nMost mentioned issues: {', '.join(top_words[:5])}"
    elif category == 'feature request':
        summary += f"\nMost requested features: {', '.join(top_words[:5])}"
    else:  # positive feedback
        summary += f"\nMost appreciated aspects: {', '.join(top_words[:5])}"

    return summary

# Function to generate word cloud
def generate_wordcloud(texts, title):
    if not texts:
        return None

    try:
        # Combine all texts
        text = ' '.join(texts)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title)
        ax.axis('off')

        return fig
    except Exception as e:
        st.warning(f"Word cloud generation failed: {str(e)}. Creating simple bar chart instead.")
        try:
            # Fallback to simple bar chart of word frequencies
            # Count word frequencies
            word_freq = {}
            for text in texts:
                if isinstance(text, str):
                    words = text.lower().split()
                    for word in words:
                        if len(word) > st.session_state.config['min_word_length']:
                            word_freq[word] = word_freq.get(word, 0) + 1

            # Sort and get top words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_words = sorted_words[:15]  # Get top 15 words

            if not top_words:
                return None

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            words = [w[0] for w in top_words]
            counts = [w[1] for w in top_words]

            ax.barh(words, counts)
            ax.set_title(title)
            ax.set_xlabel('Frequency')

            return fig
        except:
            return None

# Function to create PDF report
def create_pdf_report(data, summaries, topics=None):
    try:
        # Create PDF object
        pdf = FPDF()
        pdf.add_page()

        # Set font
        pdf.set_font("Arial", size=12)

        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Product Pulse: Feedback Analysis Report", ln=True, align='C')
        pdf.ln(5)

        # Add date
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)

        # Add summaries
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Feedback Summaries", ln=True)
        pdf.ln(5)

        for category, summary in summaries.items():
            if category != 'uncategorized' or len(data[data['category'] == 'uncategorized']) > 0:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, f"{category.title()}", ln=True)
                pdf.set_font("Arial", size=10)

                # Split summary into lines to fit PDF width
                summary_lines = summary.split('\n')
                for line in summary_lines:
                    pdf.multi_cell(0, 5, line)

                pdf.ln(5)

        # Add topics if available
        if topics:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Key Topics", ln=True)
            pdf.ln(5)

            for category, category_topics in topics.items():
                if category != 'uncategorized' or len(data[data['category'] == 'uncategorized']) > 0:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, f"{category.title()} Topics", ln=True)
                    pdf.set_font("Arial", size=10)

                    for i, topic_words in enumerate(category_topics):
                        topic_text = f"Topic {i+1}: {', '.join(topic_words)}"
                        pdf.multi_cell(0, 5, topic_text)

                    pdf.ln(5)
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        # Create a simple text-based report as fallback
        report = "PRODUCT PULSE: FEEDBACK ANALYSIS REPORT\n\n"
        report += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "FEEDBACK SUMMARIES:\n\n"

        for category, summary in summaries.items():
            report += f"{category.upper()}:\n{summary}\n\n"

        # Convert to PDF using a simpler approach
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        # Split the report into lines and add to PDF
        for line in report.split('\n'):
            pdf.multi_cell(0, 5, line)

        # Add statistics
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Feedback Statistics", ln=True)
        pdf.ln(5)

        # Category distribution
        category_counts = data['category'].value_counts()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Category Distribution", ln=True)
        pdf.set_font("Arial", size=10)

        for category, count in category_counts.items():
            pdf.cell(100, 10, f"{category.title()}: {count} ({count/len(data)*100:.1f}%)", ln=True)

        pdf.ln(5)

        # Sentiment distribution
        sentiment_counts = data['sentiment'].value_counts()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Sentiment Distribution", ln=True)
        pdf.set_font("Arial", size=10)

        for sentiment, count in sentiment_counts.items():
            pdf.cell(100, 10, f"{sentiment.title()}: {count} ({count/len(data)*100:.1f}%)", ln=True)

        pdf.ln(5)

        # Platform distribution if available
        if 'platform' in data.columns:
            platform_counts = data['platform'].value_counts()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Platform Distribution", ln=True)
            pdf.set_font("Arial", size=10)

            for platform, count in platform_counts.items():
                pdf.cell(100, 10, f"{platform}: {count} ({count/len(data)*100:.1f}%)", ln=True)

        # Add raw data
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Raw Feedback Data (Sample)", ln=True)
        pdf.ln(5)

        # Table header
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(100, 10, "Text", 1)
        pdf.cell(30, 10, "Category", 1)
        pdf.cell(30, 10, "Sentiment", 1)
        pdf.cell(30, 10, "Platform", 1)
        pdf.ln()

        # Table data (first 20 rows)
        pdf.set_font("Arial", size=8)
        for i, (_, row) in enumerate(data.iterrows()):
            if i >= 20:  # Limit to 20 rows
                break

            # Truncate text if too long
            text = row['text']
            if len(text) > 50:
                text = text[:47] + "..."

            pdf.cell(100, 10, text, 1)
            pdf.cell(30, 10, row['category'], 1)
            pdf.cell(30, 10, row['sentiment'], 1)
            pdf.cell(30, 10, row['platform'], 1)
            pdf.ln()

    # Output PDF as bytes
    return pdf.output(dest='S').encode('latin1')

# Function to create Excel report
def create_excel_report(data, summaries):
    try:
        # Create a BytesIO object
        output = io.BytesIO()

        # Create a Pandas Excel writer using the BytesIO object
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write summary sheet
            summary_data = pd.DataFrame({
                'Category': list(summaries.keys()),
                'Summary': list(summaries.values())
            })
            summary_data.to_excel(writer, sheet_name='Summary', index=False)

            # Write statistics sheet
            category_counts = data['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']

            sentiment_counts = data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            # Write category counts
            category_counts.to_excel(writer, sheet_name='Statistics', startrow=0, index=False)

            # Write sentiment counts
            sentiment_counts.to_excel(writer, sheet_name='Statistics', startrow=len(category_counts) + 3, index=False)

            # Write platform counts if available
            if 'platform' in data.columns:
                platform_counts = data['platform'].value_counts().reset_index()
                platform_counts.columns = ['Platform', 'Count']
                platform_counts.to_excel(writer, sheet_name='Statistics',
                                        startrow=len(category_counts) + len(sentiment_counts) + 6,
                                        index=False)

            # Write raw data
            data.to_excel(writer, sheet_name='Raw Data', index=False)

        # Return the BytesIO object
        output.seek(0)
        return output

    except Exception as e:
        st.error(f"Error creating Excel report: {str(e)}")
        # Create a simple CSV as fallback
        try:
            output = io.BytesIO()

            # Create a simple CSV with the most important data
            data.to_csv(output, index=False)

            output.seek(0)
            return output
        except:
            # If all else fails, return an empty BytesIO
            return io.BytesIO()

# Function to create a download link
def get_download_link(file_bytes, filename, file_type):
    b64 = base64.b64encode(file_bytes).decode()

    if file_type == 'csv':
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {file_type.upper()} Report</a>'
    elif file_type == 'excel':
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {file_type.upper()} Report</a>'
    elif file_type == 'pdf':
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download {file_type.upper()} Report</a>'
    else:
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {file_type.upper()} Report</a>'

    return href

# Main app
def main():
    st.title("Product Pulse: Advanced AI Feedback Analyzer")

    # Sidebar
    st.sidebar.header("Options")

    # Configuration section in sidebar
    with st.sidebar.expander("Analysis Configuration", expanded=False):
        st.session_state.config['sentiment_threshold'] = st.slider(
            "Sentiment Threshold", 0.0, 0.5, st.session_state.config['sentiment_threshold'], 0.05,
            help="Threshold for classifying text as positive or negative"
        )
        st.session_state.config['min_topic_words'] = st.slider(
            "Words Per Topic", 3, 10, st.session_state.config['min_topic_words'], 1,
            help="Number of words to show for each topic"
        )
        st.session_state.config['num_topics'] = st.slider(
            "Number of Topics", 2, 10, st.session_state.config['num_topics'], 1,
            help="Number of topics to extract from feedback"
        )
        st.session_state.config['num_clusters'] = st.slider(
            "Number of Clusters", 2, 10, st.session_state.config['num_clusters'], 1,
            help="Number of clusters for feedback grouping"
        )
        st.session_state.config['min_word_length'] = st.slider(
            "Minimum Word Length", 2, 6, st.session_state.config['min_word_length'], 1,
            help="Minimum length of words to include in analysis"
        )

    # File upload section
    st.sidebar.header("Data Input")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload feedback data", type=["csv", "json", "txt", "xlsx", "xls"])

    # Column mapping for non-standard files
    with st.sidebar.expander("Column Mapping", expanded=False):
        st.info("If your file doesn't have standard column names, map them here:")
        text_column = st.text_input("Text Column Name", "text")
        platform_column = st.text_input("Platform Column Name", "platform")
        date_column = st.text_input("Date Column Name (optional)", "date")

    if uploaded_file is not None:
        try:
            # Show progress
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()

            # Step 1: Load data
            status_text.text("Loading data...")
            progress_bar.progress(10)

            # Determine file type and load accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv' or file_extension == 'txt':
                # For CSV files, detect encoding and delimiter
                file_content = uploaded_file.read()

                # Check if this is a Borderlands format file
                if is_borderlands_format(file_content):
                    st.info("Detected Borderlands format. Using special handler.")

                    # Save the file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                        temp_file.write(file_content)
                        temp_path = temp_file.name

                    # Process with the special handler
                    try:
                        data = process_borderlands_format(temp_path)
                        # Clean up the temporary file
                        os.unlink(temp_path)
                    except Exception as e:
                        st.error(f"Error processing Borderlands format: {str(e)}")
                        # Clean up the temporary file
                        os.unlink(temp_path)
                        return

                else:
                    # Standard CSV processing
                    encoding = detect_encoding(file_content)
                    delimiter = detect_delimiter(file_content.decode(encoding))

                    # Reset file pointer
                    uploaded_file.seek(0)

                    # Try to read with detected parameters
                    try:
                        # First, try to read with headers
                        data = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)

                        # Check if the headers look like data (no text/feedback column)
                        if "text" not in data.columns and text_column not in data.columns:
                            # Check if any column contains text that looks like feedback
                            found_text_column = False
                            for col in data.columns:
                                # Check if column values are strings and have reasonable length
                                if data[col].dtype == 'object' and data[col].astype(str).str.len().mean() > 20:
                                    data = data.rename(columns={col: "text"})
                                    found_text_column = True
                                    st.info(f"Automatically identified column '{col}' as the text column.")
                                    break

                            # If still no text column found, try reading without headers
                            if not found_text_column:
                                uploaded_file.seek(0)
                                # Try reading without headers
                                data = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding,
                                                header=None)

                                # Assign column names based on position
                                if len(data.columns) >= 4:  # Assuming format like: ID, Game, Sentiment, Text
                                    data.columns = ['id', 'platform', 'sentiment', 'text'] + [f'col_{i}' for i in range(4, len(data.columns))]
                                elif len(data.columns) == 3:
                                    data.columns = ['id', 'platform', 'text']
                                elif len(data.columns) == 2:
                                    data.columns = ['platform', 'text']
                                else:
                                    data.columns = ['text']

                                st.info("File appears to have no headers. Assigned column names based on position.")

                    except Exception as e:
                        st.warning(f"Error reading CSV with detected parameters: {str(e)}. Trying alternative methods...")

                        # If that fails, try with default parameters
                        uploaded_file.seek(0)
                        try:
                            data = pd.read_csv(uploaded_file)
                        except:
                            # If that also fails, try reading without headers
                            uploaded_file.seek(0)
                            try:
                                data = pd.read_csv(uploaded_file, header=None)
                                # Assign column names based on position
                                if len(data.columns) >= 4:  # Assuming format like: ID, Game, Sentiment, Text
                                    data.columns = ['id', 'platform', 'sentiment', 'text'] + [f'col_{i}' for i in range(4, len(data.columns))]
                                elif len(data.columns) == 3:
                                    data.columns = ['id', 'platform', 'text']
                                elif len(data.columns) == 2:
                                    data.columns = ['platform', 'text']
                                else:
                                    data.columns = ['text']

                                st.info("File appears to have no headers. Assigned column names based on position.")
                            except Exception as e2:
                                # Last resort: try to parse the file line by line
                                uploaded_file.seek(0)
                                try:
                                    lines = uploaded_file.read().decode('utf-8').splitlines()
                                    # Try to extract text from each line
                                    texts = []
                                    for line in lines:
                                        parts = line.split(',')
                                        if len(parts) >= 4:  # Assuming ID, Game, Sentiment, Text format
                                            texts.append(parts[3])
                                        else:
                                            texts.append(line)

                                    data = pd.DataFrame({'text': texts})
                                    st.warning("Used fallback parsing method. Data may not be correctly formatted.")
                                except Exception as e3:
                                    st.error(f"Failed to read CSV file: {str(e3)}")
                                    return

            elif file_extension == 'json':
                try:
                    data = pd.read_json(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading JSON file: {str(e)}")
                    return

            elif file_extension in ['xlsx', 'xls']:
                try:
                    data = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    return

            else:
                st.error(f"Unsupported file type: {file_extension}")
                return

            # Display the raw data and allow user to select columns
            st.sidebar.subheader("Raw Data Preview")
            st.sidebar.dataframe(data.head(3))

            # Step 2: Map columns if needed
            status_text.text("Mapping columns...")
            progress_bar.progress(20)

            # Allow user to select which column contains the text
            if "text" not in data.columns:
                text_column_options = list(data.columns)
                selected_text_column = st.sidebar.selectbox(
                    "Select the column containing feedback text:",
                    text_column_options,
                    key="text_column_selector"
                )

                if selected_text_column:
                    data = data.rename(columns={selected_text_column: "text"})
                    st.sidebar.success(f"Mapped '{selected_text_column}' to 'text'")

            # Check if data has other required columns or map them
            if platform_column != "platform" and platform_column in data.columns:
                data = data.rename(columns={platform_column: "platform"})

            if date_column != "date" and date_column in data.columns:
                data = data.rename(columns={date_column: "date"})

            # If we have a sentiment column, map it
            if "sentiment" not in data.columns and "sentiment" in [col.lower() for col in data.columns]:
                for col in data.columns:
                    if col.lower() == "sentiment":
                        data = data.rename(columns={col: "sentiment"})
                        break

            # Verify required columns exist
            if "text" not in data.columns:
                # If we still don't have a text column, let's create one from the first column that might contain text
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data = data.rename(columns={col: "text"})
                        st.info(f"Using column '{col}' as the text column.")
                        break

                # If we still don't have a text column, show error
                if "text" not in data.columns:
                    st.error(f"Could not find text column. Available columns: {', '.join(data.columns)}")
                    st.error("Please select a column containing feedback text.")
                    return

            # Add platform column if it doesn't exist
            if "platform" not in data.columns:
                data["platform"] = "Unknown"

            # Store raw data
            st.session_state.data = data

            # Display data preview
            st.sidebar.subheader("Data Preview")
            st.sidebar.dataframe(data.head(3))

            # Process data button
            if st.sidebar.button("Process Feedback"):
                # Step 3: Preprocess text
                status_text.text("Preprocessing text...")
                progress_bar.progress(30)

                # Create a copy of the data for processing
                processed_data = data.copy()

                # Preprocess text
                processed_data['processed_text'] = processed_data['text'].apply(
                    lambda x: preprocess_text(x, remove_stopwords=False)
                )

                # Clean text for analysis
                processed_data['clean_text'] = processed_data['text'].apply(
                    lambda x: preprocess_text(x, remove_stopwords=True)
                )

                # Step 4: Analyze sentiment
                status_text.text("Analyzing sentiment...")
                progress_bar.progress(50)

                processed_data['sentiment'] = processed_data['processed_text'].apply(analyze_sentiment)

                # Step 5: Categorize feedback
                status_text.text("Categorizing feedback...")
                progress_bar.progress(60)

                processed_data['category'] = processed_data.apply(
                    lambda row: categorize_feedback(row['processed_text'], row['sentiment']),
                    axis=1
                )

                # Step 6: Cluster feedback
                status_text.text("Clustering feedback...")
                progress_bar.progress(70)

                # Get clean texts for clustering
                clean_texts = processed_data['clean_text'].tolist()

                # Perform clustering
                clusters = cluster_feedback(clean_texts, st.session_state.config['num_clusters'])
                processed_data['cluster'] = clusters

                # Store processed data
                st.session_state.processed_data = processed_data

                # Step 7: Generate summaries
                status_text.text("Generating summaries...")
                progress_bar.progress(80)

                # Generate summaries for each category
                categories = ['pain point', 'feature request', 'positive feedback', 'uncategorized']
                summaries = {}

                for category in categories:
                    # Get texts for this category
                    category_data = processed_data[processed_data['category'] == category]
                    category_texts = category_data['clean_text'].tolist()

                    # Get sentiment counts for this category
                    sentiment_counts = category_data['sentiment'].value_counts().to_dict()

                    # Generate summary
                    summaries[category] = summarize_feedback(category_texts, category, sentiment_counts)

                # Store summaries
                st.session_state.summaries = summaries

                # Complete
                status_text.text("Processing complete!")
                progress_bar.progress(100)

                # Clear progress after a delay
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

                st.sidebar.success("Feedback processed successfully!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

    # Main content
    if st.session_state.processed_data is not None:
        # Display tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Sentiment Analysis", "Categories", "Topics & Clusters", "Export"
        ])

        with tab1:
            st.header("Feedback Overview")

            # Display basic stats
            data = st.session_state.processed_data

            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Feedback", len(data))

            with col2:
                positive_pct = len(data[data['sentiment'] == 'positive']) / len(data) * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")

            with col3:
                pain_pct = len(data[data['category'] == 'pain point']) / len(data) * 100
                st.metric("Pain Points", f"{pain_pct:.1f}%")

            with col4:
                feature_pct = len(data[data['category'] == 'feature request']) / len(data) * 100
                st.metric("Feature Requests", f"{feature_pct:.1f}%")

            # Display platform distribution if available
            if 'platform' in data.columns and len(data['platform'].unique()) > 1:
                st.subheader("Feedback by Platform")

                # Create a pie chart
                platform_counts = data['platform'].value_counts()
                fig = px.pie(
                    values=platform_counts.values,
                    names=platform_counts.index,
                    title="Feedback Distribution by Platform"
                )
                st.plotly_chart(fig)

            # Display raw data
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(st.session_state.data)

        with tab2:
            st.header("Sentiment Analysis")

            data = st.session_state.processed_data

            # Sentiment distribution
            st.subheader("Sentiment Distribution")

            # Create columns
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display counts
                sentiment_counts = data['sentiment'].value_counts()
                st.dataframe(sentiment_counts)

            with col2:
                # Create a pie chart
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
                )
                st.plotly_chart(fig)

            # Sentiment by platform if available
            if 'platform' in data.columns and len(data['platform'].unique()) > 1:
                st.subheader("Sentiment by Platform")

                # Create a grouped bar chart
                platform_sentiment = pd.crosstab(data['platform'], data['sentiment'])

                # Convert to percentage
                platform_sentiment_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100

                # Create the chart
                fig = px.bar(
                    platform_sentiment_pct.reset_index().melt(id_vars='platform'),
                    x='platform',
                    y='value',
                    color='sentiment',
                    title="Sentiment Distribution by Platform (%)",
                    labels={'value': 'Percentage', 'platform': 'Platform'},
                    color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
                )
                st.plotly_chart(fig)

            # Word clouds by sentiment
            st.subheader("Word Clouds by Sentiment")

            # Create columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Positive Feedback")
                positive_texts = data[data['sentiment'] == 'positive']['clean_text'].tolist()
                if positive_texts:
                    fig = generate_wordcloud(positive_texts, "Positive Feedback")
                    st.pyplot(fig)
                else:
                    st.info("No positive feedback available")

            with col2:
                st.write("Neutral Feedback")
                neutral_texts = data[data['sentiment'] == 'neutral']['clean_text'].tolist()
                if neutral_texts:
                    fig = generate_wordcloud(neutral_texts, "Neutral Feedback")
                    st.pyplot(fig)
                else:
                    st.info("No neutral feedback available")

            with col3:
                st.write("Negative Feedback")
                negative_texts = data[data['sentiment'] == 'negative']['clean_text'].tolist()
                if negative_texts:
                    fig = generate_wordcloud(negative_texts, "Negative Feedback")
                    st.pyplot(fig)
                else:
                    st.info("No negative feedback available")

        with tab3:
            st.header("Feedback Categories")

            data = st.session_state.processed_data

            # Category distribution
            st.subheader("Category Distribution")

            # Create columns
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display counts
                category_counts = data['category'].value_counts()
                st.dataframe(category_counts)

            with col2:
                # Create a pie chart
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Category Distribution"
                )
                st.plotly_chart(fig)

            # Display summaries
            st.subheader("Category Summaries")

            for category, summary in st.session_state.summaries.items():
                if category != 'uncategorized' or len(data[data['category'] == 'uncategorized']) > 0:
                    with st.expander(f"{category.title()} Summary", expanded=True):
                        st.markdown(summary)

            # Display categorized data
            st.subheader("Categorized Data")

            # Add filter by category
            selected_category = st.selectbox(
                "Filter by Category",
                ['All'] + list(data['category'].unique())
            )

            if selected_category == 'All':
                filtered_data = data
            else:
                filtered_data = data[data['category'] == selected_category]

            st.dataframe(filtered_data[['text', 'platform', 'category', 'sentiment']])

        with tab4:
            st.header("Topics & Clusters")

            data = st.session_state.processed_data

            # Display topics
            st.subheader("Key Topics by Category")

            for category, topics in st.session_state.topics.items():
                if category != 'uncategorized' or len(data[data['category'] == 'uncategorized']) > 0:
                    with st.expander(f"{category.title()} Topics", expanded=True):
                        for i, topic_words in enumerate(topics):
                            st.write(f"Topic {i+1}: {', '.join(topic_words)}")

            # Display clusters
            st.subheader("Feedback Clusters")

            # Get cluster counts
            cluster_counts = data['cluster'].value_counts().sort_index()

            # Create columns
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(cluster_counts)

            with col2:
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    title="Feedback Distribution by Cluster",
                    labels={'x': 'Cluster', 'y': 'Count'}
                )
                st.plotly_chart(fig)

            # Show top words for each cluster
            st.subheader("Top Words by Cluster")

            for cluster_id in sorted(data['cluster'].unique()):
                cluster_texts = data[data['cluster'] == cluster_id]['clean_text'].tolist()

                if cluster_texts:
                    # Count word frequency
                    word_freq = {}
                    for text in cluster_texts:
                        if isinstance(text, str):
                            words = text.split()
                            for word in words:
                                if len(word) > st.session_state.config['min_word_length']:
                                    word_freq[word] = word_freq.get(word, 0) + 1

                    # Sort words by frequency
                    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                    top_words = [f"{w[0]} ({w[1]})" for w in sorted_words[:10] if w[1] > 1]

                    st.write(f"Cluster {cluster_id}: {', '.join(top_words)}")

            # Display cluster data
            with st.expander("View Cluster Data", expanded=False):
                # Add filter by cluster
                selected_cluster = st.selectbox(
                    "Filter by Cluster",
                    ['All'] + sorted(list(data['cluster'].unique()))
                )

                if selected_cluster == 'All':
                    cluster_data = data
                else:
                    cluster_data = data[data['cluster'] == selected_cluster]

                st.dataframe(cluster_data[['text', 'category', 'sentiment', 'cluster']])

        with tab5:
            st.header("Export Reports")

            # Create columns for different export options
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("CSV Export")
                if st.button("Generate CSV Report"):
                    with st.spinner("Generating CSV..."):
                        # Create a report DataFrame
                        report_df = pd.DataFrame({
                            'Category': list(st.session_state.summaries.keys()),
                            'Summary': list(st.session_state.summaries.values()),
                            'Count': [len(st.session_state.processed_data[st.session_state.processed_data['category'] == cat]) for cat in st.session_state.summaries.keys()]
                        })

                        # Convert to CSV
                        csv = report_df.to_csv(index=False).encode()

                        # Create download link
                        st.markdown(get_download_link(csv, "feedback_analysis.csv", "csv"), unsafe_allow_html=True)

            with col2:
                st.subheader("Excel Export")
                if st.button("Generate Excel Report"):
                    with st.spinner("Generating Excel..."):
                        # Create Excel report
                        excel_bytes = create_excel_report(st.session_state.processed_data, st.session_state.summaries)

                        # Create download link
                        st.markdown(get_download_link(excel_bytes.getvalue(), "feedback_analysis.xlsx", "excel"), unsafe_allow_html=True)

            with col3:
                st.subheader("PDF Export")
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF..."):
                        # Create PDF report
                        pdf_bytes = create_pdf_report(
                            st.session_state.processed_data,
                            st.session_state.summaries,
                            st.session_state.topics
                        )

                        # Create download link
                        st.markdown(get_download_link(pdf_bytes, "feedback_analysis.pdf", "pdf"), unsafe_allow_html=True)

            # Export raw data
            st.subheader("Export Raw Data")
            if st.button("Export Processed Data"):
                with st.spinner("Preparing data..."):
                    # Convert to CSV
                    csv = st.session_state.processed_data.to_csv(index=False).encode()

                    # Create download link
                    st.markdown(get_download_link(csv, "processed_feedback_data.csv", "csv"), unsafe_allow_html=True)

    else:
        # Display instructions
        st.info("""
        ## How to use Product Pulse Advanced:

        1. Upload a feedback data file (CSV, JSON, Excel, or text)
        2. If your file has non-standard column names, map them in the "Column Mapping" section
        3. Click "Process Feedback" to analyze the data
        4. Explore the results in the tabs and generate reports

        ### Advanced Features:

        - **Sentiment Analysis**: Automatically detects positive, neutral, and negative feedback
        - **Topic Modeling**: Identifies key themes in your feedback
        - **Clustering**: Groups similar feedback together
        - **Multiple Export Options**: Generate CSV, Excel, or PDF reports
        - **Customizable Analysis**: Adjust parameters in the configuration panel
        """)

        # Sample data format
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame({
            'text': [
                "I love this app! It's so intuitive and helpful.",
                "The app keeps crashing when I try to upload photos.",
                "Would be great if you could add dark mode to the app."
            ],
            'platform': ['App Store', 'Google Play', 'Twitter'],
            'date': ['2023-01-15', '2023-01-20', '2023-01-25']
        })

        st.dataframe(sample_data)

        # Option to generate sample data
        if st.button("Generate Sample Data File"):
            # Create more sample data
            texts = [
                "I love this app! It's so intuitive and helpful.",
                "The app keeps crashing when I try to upload photos.",
                "Would be great if you could add dark mode to the app.",
                "This is the best productivity app I've ever used!",
                "Can't login after the latest update. Please fix ASAP.",
                "The UI is beautiful but navigation is confusing.",
                "Would love to see integration with other tools.",
                "App is too slow on older devices.",
                "Customer support was amazing when I had an issue.",
                "The new feature is exactly what I was looking for!",
                "I've been using this app for years and it keeps getting better!",
                "The notifications are too frequent and annoying.",
                "Would it be possible to add multi-user support?",
                "The app crashes every time I try to save my progress.",
                "Please add the ability to export data to CSV.",
                "The search functionality is broken in the latest update.",
                "I'm impressed with how responsive the app is on my old phone.",
                "The pricing is too high compared to similar apps.",
                "Your recent update fixed all the issues I was having!",
                "The app needs better documentation for new users."
            ]

            platforms = ['App Store', 'Google Play', 'Twitter', 'Email', 'Website']

            # Generate dates in the last 90 days
            today = datetime.datetime.now()
            dates = [(today - datetime.timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d') for _ in range(30)]

            # Create DataFrame
            sample_df = pd.DataFrame({
                'text': np.random.choice(texts, size=30, replace=True),
                'platform': np.random.choice(platforms, size=30, replace=True),
                'date': dates
            })

            # Convert to CSV
            csv = sample_df.to_csv(index=False).encode()

            # Create download link
            st.markdown(get_download_link(csv, "sample_feedback.csv", "csv"), unsafe_allow_html=True)

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    # Make sure session state is initialized
    initialize_session_state()

    # Sidebar
    st.sidebar.header("Options")

    # Configuration panel
    with st.sidebar.expander("Configuration", expanded=False):
        # Update config values based on sliders
        st.session_state.config['sentiment_threshold'] = st.slider(
            "Sentiment Threshold", 0.0, 0.5, st.session_state.config['sentiment_threshold'], 0.05,
            help="Threshold for classifying sentiment as positive or negative"
        )
        st.session_state.config['min_topic_words'] = st.slider(
            "Minimum Topic Words", 3, 10, st.session_state.config['min_topic_words'], 1,
            help="Minimum number of words to include in each topic"
        )
        st.session_state.config['num_topics'] = st.slider(
            "Number of Topics", 2, 10, st.session_state.config['num_topics'], 1,
            help="Number of topics to extract from feedback"
        )
        st.session_state.config['num_clusters'] = st.slider(
            "Number of Clusters", 2, 10, st.session_state.config['num_clusters'], 1,
            help="Number of clusters to create from feedback"
        )
        st.session_state.config['min_word_length'] = st.slider(
            "Minimum Word Length", 2, 6, st.session_state.config['min_word_length'], 1,
            help="Minimum length of words to include in analysis"
        )

    if uploaded_file is not None:
        try:
            # Process the file based on its type
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # Read the file content
            file_content = uploaded_file.read()

            # Check if it's a Borderlands format
            if is_borderlands_format(file_content):
                st.info("Detected Borderlands format. Using special handler.")
                # Reset the file pointer
                uploaded_file.seek(0)
                data = process_borderlands_format(uploaded_file)
            else:
                # Reset the file pointer
                uploaded_file.seek(0)

                if file_extension == 'csv':
                    # Detect delimiter and encoding
                    delimiter = detect_delimiter(file_content.decode('utf-8', errors='ignore'))
                    encoding = detect_encoding(file_content)

                    # Read CSV with detected parameters
                    try:
                        # For newer versions of pandas (1.3.0+)
                        data = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding, on_bad_lines='warn')
                    except TypeError:
                        # For older versions of pandas
                        data = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding, error_bad_lines=False)

                elif file_extension == 'json':
                    data = pd.read_json(uploaded_file)

                elif file_extension == 'xlsx':
                    data = pd.read_excel(uploaded_file)

                elif file_extension == 'txt':
                    # For text files, create a simple DataFrame with one column
                    lines = file_content.decode('utf-8', errors='ignore').splitlines()
                    data = pd.DataFrame({'text': [line for line in lines if line.strip()]})

                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return

            # Display column mapping if needed
            if 'text' not in data.columns:
                st.warning("Text column not found. Please map columns.")

                # Display column mapping interface
                st.subheader("Column Mapping")

                # Map text column
                text_col = st.selectbox("Select the column containing feedback text:", data.columns)

                # Map platform column if available
                platform_col = None
                if len(data.columns) > 1:
                    platform_options = ['None'] + list(data.columns)
                    platform_col = st.selectbox("Select the column containing platform information (optional):", platform_options)
                    if platform_col == 'None':
                        platform_col = None

                # Apply mapping
                if st.button("Apply Mapping"):
                    # Create a new DataFrame with mapped columns
                    mapped_data = pd.DataFrame()
                    mapped_data['text'] = data[text_col]

                    if platform_col:
                        mapped_data['platform'] = data[platform_col]
                    else:
                        mapped_data['platform'] = 'Unknown'

                    data = mapped_data
                    st.success("Column mapping applied!")
                else:
                    return

            # Ensure platform column exists
            if 'platform' not in data.columns:
                data['platform'] = 'Unknown'

            # Store the data
            st.session_state.data = data

            # Process data button
            if st.sidebar.button("Process Feedback"):
                with st.spinner("Processing feedback..."):
                    # Preprocess text
                    data['processed_text'] = data['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

                    # Analyze sentiment
                    data['sentiment'] = data['processed_text'].apply(analyze_sentiment)

                    # Categorize feedback
                    data['category'] = data.apply(lambda row: categorize_feedback(row['processed_text'], row['sentiment']), axis=1)

                    # Store processed data
                    st.session_state.processed_data = data

                    # Generate summaries for each category
                    categories = ['pain point', 'feature request', 'positive feedback', 'uncategorized']
                    summaries = {}

                    for category in categories:
                        category_texts = data[data['category'] == category]['processed_text'].tolist()

                        # Get sentiment counts for this category
                        sentiment_counts = data[data['category'] == category]['sentiment'].value_counts().to_dict()

                        # Generate summary
                        summaries[category] = summarize_feedback(category_texts, category, sentiment_counts)

                    st.session_state.summaries = summaries

                    # Cluster the feedback
                    all_texts = data['processed_text'].tolist()
                    clusters = cluster_feedback(all_texts, st.session_state.config['num_clusters'])
                    data['cluster'] = clusters

                    # Update processed data with clusters
                    st.session_state.processed_data = data

                st.success("Feedback processed successfully!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

    # Main content
    if st.session_state.processed_data is not None:
        # Display tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Sentiment Analysis", "Categories", "Topics & Clusters", "Export"])

        with tab1:
            st.header("Feedback Overview")

            # Display basic stats
            st.subheader("Basic Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Feedback Items", len(st.session_state.processed_data))

            with col2:
                platforms = st.session_state.processed_data['platform'].value_counts()
                st.metric("Platforms", len(platforms))

            with col3:
                avg_length = st.session_state.processed_data['text'].str.len().mean()
                st.metric("Average Feedback Length", f"{avg_length:.1f} chars")

            # Display raw data
            st.subheader("Raw Data")
            st.dataframe(st.session_state.data)

        with tab2:
            st.header("Sentiment Analysis")

            # Display sentiment distribution
            sentiment_counts = st.session_state.processed_data['sentiment'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Sentiment Counts")
                st.dataframe(sentiment_counts)

            with col2:
                st.subheader("Sentiment Distribution")

                # Create a pie chart with Plotly
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': 'green',
                        'neutral': 'gray',
                        'negative': 'red'
                    }
                )
                st.plotly_chart(fig)

            # Display sentiment by platform
            st.subheader("Sentiment by Platform")

            # Create a grouped bar chart
            sentiment_by_platform = pd.crosstab(
                st.session_state.processed_data['platform'],
                st.session_state.processed_data['sentiment']
            )

            fig = px.bar(
                sentiment_by_platform,
                barmode='group',
                title="Sentiment by Platform",
                color_discrete_map={
                    'positive': 'green',
                    'neutral': 'gray',
                    'negative': 'red'
                }
            )
            st.plotly_chart(fig)

            # Display word clouds by sentiment
            st.subheader("Word Clouds by Sentiment")

            sentiment_tabs = st.tabs(["Positive", "Neutral", "Negative"])

            with sentiment_tabs[0]:
                positive_texts = st.session_state.processed_data[st.session_state.processed_data['sentiment'] == 'positive']['processed_text'].tolist()
                positive_cloud = generate_wordcloud(positive_texts, "Positive Feedback Word Cloud")
                if positive_cloud:
                    st.pyplot(positive_cloud)
                else:
                    st.info("Not enough positive feedback for word cloud.")

            with sentiment_tabs[1]:
                neutral_texts = st.session_state.processed_data[st.session_state.processed_data['sentiment'] == 'neutral']['processed_text'].tolist()
                neutral_cloud = generate_wordcloud(neutral_texts, "Neutral Feedback Word Cloud")
                if neutral_cloud:
                    st.pyplot(neutral_cloud)
                else:
                    st.info("Not enough neutral feedback for word cloud.")

            with sentiment_tabs[2]:
                negative_texts = st.session_state.processed_data[st.session_state.processed_data['sentiment'] == 'negative']['processed_text'].tolist()
                negative_cloud = generate_wordcloud(negative_texts, "Negative Feedback Word Cloud")
                if negative_cloud:
                    st.pyplot(negative_cloud)
                else:
                    st.info("Not enough negative feedback for word cloud.")

        with tab3:
            st.header("Feedback Categories")

            # Display category distribution
            category_counts = st.session_state.processed_data['category'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Category Counts")
                st.dataframe(category_counts)

            with col2:
                st.subheader("Category Distribution")

                # Create a pie chart with Plotly
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Category Distribution",
                    color=category_counts.index,
                    color_discrete_map={
                        'pain point': 'red',
                        'feature request': 'blue',
                        'positive feedback': 'green',
                        'uncategorized': 'gray'
                    }
                )
                st.plotly_chart(fig)

            # Display summaries
            st.subheader("Category Summaries")

            for category, summary in st.session_state.summaries.items():
                if category != 'uncategorized' or len(st.session_state.processed_data[st.session_state.processed_data['category'] == 'uncategorized']) > 0:
                    with st.expander(f"{category.title()} ({len(st.session_state.processed_data[st.session_state.processed_data['category'] == category])} items)", expanded=True):
                        st.markdown(summary)

            # Display categorized data
            st.subheader("Categorized Data")
            st.dataframe(st.session_state.processed_data[['text', 'platform', 'sentiment', 'category']])

        with tab4:
            st.header("Topics & Clusters")

            # Display topics
            st.subheader("Key Topics by Category")

            for category, topics in st.session_state.topics.items():
                if category != 'uncategorized' or len(st.session_state.processed_data[st.session_state.processed_data['category'] == 'uncategorized']) > 0:
                    with st.expander(f"Topics in {category.title()}", expanded=True):
                        for i, topic_words in enumerate(topics):
                            st.write(f"**Topic {i+1}:** {', '.join(topic_words)}")

            # Display clusters
            st.subheader("Feedback Clusters")

            # Count feedback items per cluster
            cluster_counts = st.session_state.processed_data['cluster'].value_counts().sort_index()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("Cluster Sizes")
                st.dataframe(cluster_counts)

            with col2:
                # Create a bar chart with Plotly
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Count'},
                    title="Feedback Items per Cluster"
                )
                st.plotly_chart(fig)

            # Display top words for each cluster
            st.subheader("Top Words by Cluster")

            for cluster in sorted(st.session_state.processed_data['cluster'].unique()):
                cluster_texts = st.session_state.processed_data[st.session_state.processed_data['cluster'] == cluster]['processed_text'].tolist()

                # Count word frequency
                word_freq = {}
                for text in cluster_texts:
                    words = text.split()
                    for word in words:
                        if len(word) > st.session_state.config['min_word_length']:
                            word_freq[word] = word_freq.get(word, 0) + 1

                # Sort words by frequency
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                top_words = [w[0] for w in sorted_words[:10] if w[1] > 1]

                with st.expander(f"Cluster {cluster} ({len(cluster_texts)} items)", expanded=False):
                    if top_words:
                        st.write(f"**Top words:** {', '.join(top_words)}")
                    else:
                        st.write("No common words found in this cluster.")

                    # Display sample feedback from this cluster
                    st.write("**Sample feedback:**")
                    samples = st.session_state.processed_data[st.session_state.processed_data['cluster'] == cluster]['text'].head(3).tolist()
                    for i, sample in enumerate(samples):
                        st.write(f"{i+1}. {sample}")

        with tab5:
            st.header("Export Results")

            # Export options
            st.subheader("Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Generate CSV Report"):
                    with st.spinner("Generating CSV..."):
                        # Create a report DataFrame
                        report_df = pd.DataFrame({
                            'Category': list(st.session_state.summaries.keys()),
                            'Summary': list(st.session_state.summaries.values()),
                            'Count': [len(st.session_state.processed_data[st.session_state.processed_data['category'] == cat]) for cat in st.session_state.summaries.keys()]
                        })

                        # Convert to CSV
                        csv = report_df.to_csv(index=False).encode()

                        # Create download link
                        st.markdown(get_download_link(csv, "feedback_analysis.csv", "csv"), unsafe_allow_html=True)

            with col2:
                if st.button("Generate Excel Report"):
                    with st.spinner("Generating Excel..."):
                        try:
                            # Create Excel file in memory
                            output = BytesIO()

                            # Create Excel writer
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # Write summary sheet
                                summary_df = pd.DataFrame({
                                    'Category': list(st.session_state.summaries.keys()),
                                    'Count': [len(st.session_state.processed_data[st.session_state.processed_data['category'] == cat]) for cat in st.session_state.summaries.keys()]
                                })
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                                # Write full data sheet
                                st.session_state.processed_data.to_excel(writer, sheet_name='Full Data', index=False)

                                # Write sheet for each category
                                for category in st.session_state.summaries.keys():
                                    if category != 'uncategorized' or len(st.session_state.processed_data[st.session_state.processed_data['category'] == 'uncategorized']) > 0:
                                        category_data = st.session_state.processed_data[st.session_state.processed_data['category'] == category]
                                        if not category_data.empty:
                                            category_data.to_excel(writer, sheet_name=category[:31], index=False)  # Excel sheet names limited to 31 chars

                            # Get the Excel data
                            excel_data = output.getvalue()

                            # Create download link
                            st.markdown(get_download_link(excel_data, "feedback_analysis.xlsx", "xlsx"), unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
                            # Fallback to CSV
                            st.warning("Falling back to CSV format.")
                            csv = st.session_state.processed_data.to_csv(index=False).encode()
                            st.markdown(get_download_link(csv, "feedback_analysis.csv", "csv"), unsafe_allow_html=True)

            with col3:
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF..."):
                        try:
                            # Create PDF report
                            pdf_data = create_pdf_report(st.session_state.processed_data, st.session_state.summaries, st.session_state.topics)

                            # Create download link
                            st.markdown(get_download_link(pdf_data, "feedback_analysis.pdf", "pdf"), unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                            # Fallback to text
                            st.warning("Falling back to text format.")

                            # Create a simple text report
                            text_report = "# Feedback Analysis Report\n\n"
                            text_report += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                            text_report += "## Summaries\n\n"
                            for category, summary in st.session_state.summaries.items():
                                text_report += f"### {category.title()}\n\n{summary}\n\n"

                            # Convert to bytes
                            text_data = text_report.encode()

                            # Create download link
                            st.markdown(get_download_link(text_data, "feedback_analysis.txt", "txt"), unsafe_allow_html=True)

            # Export raw data
            st.subheader("Export Raw Data")
            if st.button("Export Processed Data"):
                with st.spinner("Preparing data..."):
                    # Convert to CSV
                    csv = st.session_state.processed_data.to_csv(index=False).encode()

                    # Create download link
                    st.markdown(get_download_link(csv, "processed_feedback_data.csv", "csv"), unsafe_allow_html=True)

    else:
        # Display instructions
        st.info("""
        ## How to use Product Pulse Advanced:

        1. Upload a feedback data file (CSV, JSON, Excel, or text)
        2. If your file has non-standard column names, map them in the "Column Mapping" section
        3. Click "Process Feedback" to analyze the data
        4. Explore the results in the tabs and generate reports

        ### Advanced Features:

        - **Sentiment Analysis**: Automatically detects positive, neutral, and negative feedback
        - **Topic Modeling**: Identifies key themes in your feedback
        - **Clustering**: Groups similar feedback together
        - **Multiple Export Options**: Generate CSV, Excel, or PDF reports
        - **Customizable Analysis**: Adjust parameters in the configuration panel
        """)

        # Sample data format
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame({
            'text': [
                "I love this app! It's so intuitive and helpful.",
                "The app keeps crashing when I try to upload photos.",
                "Would be great if you could add dark mode to the app."
            ],
            'platform': ['App Store', 'Google Play', 'Twitter'],
            'date': ['2023-01-15', '2023-01-20', '2023-01-25']
        })

        st.dataframe(sample_data)

        # Option to generate sample data
        if st.button("Generate Sample Data File"):
            # Create more sample data
            texts = [
                "I love this app! It's so intuitive and helpful.",
                "The app keeps crashing when I try to upload photos.",
                "Would be great if you could add dark mode to the app.",
                "This is the best productivity app I've ever used!",
                "Can't login after the latest update. Please fix ASAP.",
                "The UI is beautiful but navigation is confusing.",
                "Would love to see integration with other tools.",
                "App is too slow on older devices.",
                "Customer support was amazing when I had an issue.",
                "The new feature is exactly what I was looking for!",
                "I've been using this app for years and it keeps getting better!",
                "The notifications are too frequent and annoying.",
                "Would it be possible to add multi-user support?",
                "The app crashes every time I try to save my progress.",
                "Please add the ability to export data to CSV.",
                "The search functionality is broken in the latest update.",
                "I'm impressed with how responsive the app is on my old phone.",
                "The pricing is too high compared to similar apps.",
                "Your recent update fixed all the issues I was having!",
                "The app needs better documentation for new users."
            ]

            platforms = ['App Store', 'Google Play', 'Twitter', 'Email', 'Website']

            # Generate dates in the last 90 days
            today = datetime.datetime.now()
            dates = [(today - datetime.timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d') for _ in range(30)]

            # Create DataFrame
            sample_df = pd.DataFrame({
                'text': np.random.choice(texts, size=30, replace=True),
                'platform': np.random.choice(platforms, size=30, replace=True),
                'date': dates
            })

            # Convert to CSV
            csv = sample_df.to_csv(index=False).encode()

            # Create download link
            st.markdown(get_download_link(csv, "sample_feedback.csv", "csv"), unsafe_allow_html=True)

# Main app function
def main():
    # Make sure session state is initialized
    initialize_session_state()

    # Display title
    st.title("Product Pulse: Advanced AI Feedback Analyzer")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload feedback data", type=["csv", "json", "xlsx", "txt"])

    # Process the uploaded file
    process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
