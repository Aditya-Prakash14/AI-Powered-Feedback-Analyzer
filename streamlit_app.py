import streamlit as st
import os
import sys
import subprocess
import importlib.util

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Product Pulse: Advanced AI Feedback Analyzer",
    page_icon="📊",
    layout="wide",
)

# Function to check if a module is installed
def is_module_installed(module_name):
    return importlib.util.find_spec(module_name) is not None

# Function to install a module
def install_module(module_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])

# Function to download NLTK resources
def download_nltk_resources():
    import nltk

    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            st.success(f"✅ Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            st.error(f"❌ Failed to download NLTK resource {resource}: {str(e)}")

# Function to fix punkt_tab issue
def fix_punkt_tab_issue():
    import os
    import shutil

    # Get NLTK data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')

    # Define paths
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab')

    # Check if punkt directory exists
    if not os.path.exists(punkt_dir):
        st.error(f"❌ Punkt directory not found: {punkt_dir}")
        return False

    # Check if punkt_tab directory already exists
    if os.path.exists(punkt_tab_dir):
        st.success(f"✅ Punkt_tab directory already exists: {punkt_tab_dir}")
        return True

    # Create punkt_tab directory
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(punkt_tab_dir), exist_ok=True)

        # Copy from punkt to punkt_tab
        shutil.copytree(punkt_dir, punkt_tab_dir)
        st.success(f"✅ Created punkt_tab directory: {punkt_tab_dir}")

        return True

    except Exception as e:
        st.error(f"❌ Error creating punkt_tab directory: {str(e)}")

        try:
            # If copy fails, try creating manually
            if os.path.exists(punkt_dir):
                # Create punkt_tab directory
                os.makedirs(punkt_tab_dir, exist_ok=True)

                # Create english directory inside punkt_tab
                english_dir = os.path.join(punkt_tab_dir, 'english')
                os.makedirs(english_dir, exist_ok=True)

                # Copy pickle files from punkt to punkt_tab
                punkt_english_dir = os.path.join(punkt_dir, 'english')
                if os.path.exists(punkt_english_dir):
                    for file in os.listdir(punkt_english_dir):
                        if file.endswith('.pickle'):
                            src = os.path.join(punkt_english_dir, file)
                            dst = os.path.join(english_dir, file)
                            shutil.copy2(src, dst)

                st.success(f"✅ Created punkt_tab directory manually: {punkt_tab_dir}")
                return True
            else:
                st.error(f"❌ Punkt directory not found: {punkt_dir}")
                return False

        except Exception as e2:
            st.error(f"❌ Alternative method also failed: {str(e2)}")
            return False

# Main app
def main():
    st.title("Product Pulse: Advanced AI Feedback Analyzer")

    # Check and install required packages
    required_packages = {
        'nltk': 'nltk',
        'textblob': 'textblob',
        'sklearn': 'scikit-learn',
        'wordcloud': 'wordcloud',
        'plotly': 'plotly',
        'fpdf': 'fpdf'
    }

    missing_packages = []
    for package, pip_name in required_packages.items():
        if not is_module_installed(package):
            missing_packages.append(pip_name)

    if missing_packages:
        st.warning(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            install_module(package)
        st.experimental_rerun()

    # Download NLTK resources
    st.header("Setting up NLTK Resources")

    if st.button("Download NLTK Resources"):
        download_nltk_resources()
        fix_punkt_tab_issue()
        st.success("✅ NLTK setup complete!")
        st.balloons()

    # Launch the main app
    st.header("Launch Product Pulse")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Launch Advanced App", use_container_width=True):
            # Run the advanced app
            st.session_state['app_mode'] = 'advanced'
            st.rerun()

    with col2:
        if st.button("Launch Basic App", use_container_width=True):
            # Run the simple app
            st.session_state['app_mode'] = 'simple'
            st.rerun()

    # Check if we should run one of the apps
    if 'app_mode' in st.session_state:
        if st.session_state['app_mode'] == 'advanced':
            st.subheader("Running Advanced App...")
            # Import the module but don't call set_page_config again
            import advanced_app
            # Initialize session state
            advanced_app.initialize_session_state()
            # Run the main function from the module
            advanced_app.main()
            # Add a button to return to the launcher
            if st.button("Return to Launcher"):
                del st.session_state['app_mode']
                st.rerun()
        elif st.session_state['app_mode'] == 'simple':
            st.subheader("Running Basic App...")
            # Import the module but don't call set_page_config again
            import simple_app
            # Run the main function from the module
            simple_app.main()
            # Add a button to return to the launcher
            if st.button("Return to Launcher"):
                del st.session_state['app_mode']
                st.rerun()

    # App information
    st.markdown("""
    ## About Product Pulse

    Product Pulse is an AI-powered feedback analyzer that helps you understand customer feedback from various sources.

    ### Features:
    - Upload feedback data in various formats (CSV, JSON, Excel)
    - Analyze sentiment and categorize feedback
    - Extract key topics and themes
    - Generate visual reports and summaries
    - Export results in multiple formats

    ### Getting Started:
    1. Click "Download NLTK Resources" to set up the required language processing resources
    2. Click "Launch Advanced App" to start the full-featured application
    3. Upload your feedback data and start analyzing!
    """)

if __name__ == "__main__":
    main()
