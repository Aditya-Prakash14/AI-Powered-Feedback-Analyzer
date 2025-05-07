import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Page configuration is handled by streamlit_app.py

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

# Simple rule-based categorization
def categorize_feedback(text):
    text = text.lower()

    # Define keywords for each category
    pain_points = ['crash', 'bug', 'error', 'slow', 'issue', 'problem', 'fix', 'broken', 'annoying', 'difficult']
    feature_requests = ['add', 'feature', 'would be', 'could you', 'please', 'implement', 'integration', 'support', 'option']
    positive_feedback = ['love', 'great', 'awesome', 'amazing', 'good', 'excellent', 'best', 'fantastic', 'helpful', 'intuitive']

    # Count matches for each category
    pain_count = sum(1 for word in pain_points if word in text)
    feature_count = sum(1 for word in feature_requests if word in text)
    positive_count = sum(1 for word in positive_feedback if word in text)

    # Determine the category with the most matches
    counts = [pain_count, feature_count, positive_count]
    categories = ['pain point', 'feature request', 'positive feedback']

    if max(counts) == 0:
        return 'uncategorized'

    return categories[counts.index(max(counts))]

# Simple summarization function
def summarize_feedback(texts, category):
    if not texts:
        return f"No {category} feedback identified."

    if len(texts) == 1:
        return texts[0]

    # Count word frequency
    word_freq = {}
    for text in texts:
        words = text.lower().split()
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1

    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Generate a simple summary based on the category
    if category == 'pain point':
        return f"Users reported {len(texts)} issues. Common concerns include {', '.join([w[0] for w in sorted_words[:5] if w[1] > 1])}."
    elif category == 'feature request':
        return f"Users requested {len(texts)} features. Common requests include {', '.join([w[0] for w in sorted_words[:5] if w[1] > 1])}."
    else:  # positive feedback
        return f"Users shared {len(texts)} positive comments. They particularly liked {', '.join([w[0] for w in sorted_words[:5] if w[1] > 1])}."

# Function to create a download link
def get_download_link(csv_string, filename="feedback_analysis.csv"):
    b64 = base64.b64encode(csv_string.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Report</a>'

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

                    # Categorize feedback
                    data['category'] = data['processed_text'].apply(categorize_feedback)

                    # Store processed data
                    st.session_state.processed_data = data

                    # Generate summaries for each category
                    categories = ['pain point', 'feature request', 'positive feedback', 'uncategorized']
                    summaries = {}

                    for category in categories:
                        category_texts = data[data['category'] == category]['processed_text'].tolist()
                        summaries[category] = summarize_feedback(category_texts, category)

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
                if category != 'uncategorized' or len(st.session_state.processed_data[st.session_state.processed_data['category'] == 'uncategorized']) > 0:
                    st.subheader(f"{category.title()}")
                    st.write(summary)

            # Generate CSV
            if st.button("Generate CSV Report"):
                with st.spinner("Generating CSV..."):
                    # Create a report DataFrame
                    report_df = pd.DataFrame({
                        'Category': list(st.session_state.summaries.keys()),
                        'Summary': list(st.session_state.summaries.values()),
                        'Count': [len(st.session_state.processed_data[st.session_state.processed_data['category'] == cat]) for cat in st.session_state.summaries.keys()]
                    })

                    # Convert to CSV
                    csv = report_df.to_csv(index=False)

                    # Create download link
                    st.markdown(get_download_link(csv), unsafe_allow_html=True)

    else:
        # Display instructions
        st.info("""
        ## How to use Product Pulse:
        1. Upload a CSV or JSON file with feedback data
        2. The file should have columns: 'text' and optionally 'platform'
        3. Click 'Process Feedback' to analyze the data
        4. View the results in the tabs and generate a CSV report
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
