import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Product Pulse: AI-Powered Feedback Analyzer",
    page_icon="📊",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

# Function to preprocess text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to categorize feedback
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def categorize_feedback(text, classifier):
    categories = ["pain point", "feature request", "positive feedback"]
    result = classifier(text, categories)
    return result['labels'][0]

# Function to summarize feedback
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

def summarize_feedback(texts, summarizer):
    if not texts:
        return "No feedback in this category."

    # Combine texts with separators
    combined_text = " ".join(texts)

    # Ensure the text is not too long for the model
    max_length = 512
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length]

    # Generate summary
    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to create PDF
def create_pdf(data, summaries):
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .summary {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Product Pulse: Feedback Analysis Report</h1>

        <h2>Summary of Pain Points</h2>
        <div class="summary">{summaries.get('pain point', 'No pain points identified.')}</div>

        <h2>Summary of Feature Requests</h2>
        <div class="summary">{summaries.get('feature request', 'No feature requests identified.')}</div>

        <h2>Summary of Positive Feedback</h2>
        <div class="summary">{summaries.get('positive feedback', 'No positive feedback identified.')}</div>

        <h2>Raw Feedback Data</h2>
        <table>
            <tr>
                <th>Text</th>
                <th>Platform</th>
                <th>Category</th>
            </tr>
    """

    for _, row in data.iterrows():
        html += f"""
            <tr>
                <td>{row['text']}</td>
                <td>{row['platform']}</td>
                <td>{row['category']}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    result_file = BytesIO()
    pisa.CreatePDF(html, dest=result_file)
    result_file.seek(0)
    return result_file

# Function to create a download link
def get_download_link(pdf_bytes, filename="feedback_analysis.pdf"):
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'

# Main app
def main():
    st.title("Product Pulse: AI-Powered Feedback Analyzer")

    # Sidebar
    st.sidebar.header("Options")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload feedback data (CSV or JSON)", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_json(uploaded_file)

            # Check if data has required columns
            if 'text' not in data.columns:
                st.error("Uploaded file must contain a 'text' column.")
                return

            if 'platform' not in data.columns:
                data['platform'] = 'Unknown'

            st.session_state.data = data

            # Process data button
            if st.sidebar.button("Process Feedback"):
                with st.spinner("Processing feedback..."):
                    # Preprocess text
                    data['processed_text'] = data['text'].apply(preprocess_text)

                    # Load models
                    classifier = load_classifier()
                    summarizer = load_summarizer()

                    # Categorize feedback
                    data['category'] = data['processed_text'].apply(lambda x: categorize_feedback(x, classifier))

                    # Store processed data
                    st.session_state.processed_data = data

                    # Generate summaries for each category
                    categories = ['pain point', 'feature request', 'positive feedback']
                    summaries = {}

                    for category in categories:
                        category_texts = data[data['category'] == category]['processed_text'].tolist()
                        summaries[category] = summarize_feedback(category_texts, summarizer)

                    st.session_state.summaries = summaries

                st.success("Feedback processed successfully!")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Main content
    if st.session_state.processed_data is not None:
        # Display tabs
        tab1, tab2, tab3 = st.tabs(["Raw Data", "Categorized Feedback", "Summaries"])

        with tab1:
            st.header("Raw Feedback Data")
            st.dataframe(st.session_state.data)

        with tab2:
            st.header("Categorized Feedback")

            # Display counts
            categories = st.session_state.processed_data['category'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Category Counts")
                st.dataframe(categories)

            with col2:
                st.subheader("Category Distribution")
                fig, ax = plt.subplots()
                categories.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)

            # Display categorized data
            st.subheader("Categorized Data")
            st.dataframe(st.session_state.processed_data[['text', 'platform', 'category']])

        with tab3:
            st.header("Feedback Summaries")

            for category, summary in st.session_state.summaries.items():
                st.subheader(f"{category.title()}")
                st.write(summary)

            # Generate PDF
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    pdf = create_pdf(st.session_state.processed_data, st.session_state.summaries)
                    st.markdown(get_download_link(pdf), unsafe_allow_html=True)

    else:
        # Display instructions
        st.info("""
        ## How to use Product Pulse:
        1. Upload a CSV or JSON file with feedback data
        2. The file should have columns: 'text' and optionally 'platform'
        3. Click 'Process Feedback' to analyze the data
        4. View the results in the tabs and generate a PDF report
        """)

        # Sample data format
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame({
            'text': [
                "I love this app! It's so intuitive and helpful.",
                "The app keeps crashing when I try to upload photos.",
                "Would be great if you could add dark mode to the app."
            ],
            'platform': ['App Store', 'Google Play', 'Twitter']
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
                "The new feature is exactly what I was looking for!"
            ]

            platforms = ['App Store', 'Google Play', 'Twitter', 'Email']

            # Create DataFrame
            sample_df = pd.DataFrame({
                'text': np.random.choice(texts, size=20, replace=True),
                'platform': np.random.choice(platforms, size=20, replace=True)
            })

            # Convert to CSV
            csv = sample_df.to_csv(index=False)

            # Create download link
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sample_feedback.csv">Download Sample CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
