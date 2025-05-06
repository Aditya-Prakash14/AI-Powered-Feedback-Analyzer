import nltk
import sys
import os

def download_nltk_resources():
    """
    Download required NLTK resources for the Product Pulse application.
    """
    print("Downloading NLTK resources...")

    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
        print(f"Created NLTK data directory: {nltk_data_dir}")

    resources = [
        'punkt',
        'stopwords',
        'wordnet'
    ]

    for resource in resources:
        print(f"Downloading {resource}...")
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

    # Fix for punkt_tab issue - create a symlink if needed
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab')

    if os.path.exists(punkt_dir) and not os.path.exists(punkt_tab_dir):
        try:
            # Create the punkt_tab directory
            os.makedirs(os.path.dirname(punkt_tab_dir), exist_ok=True)

            # Create a symlink or copy from punkt to punkt_tab
            if os.name == 'nt':  # Windows
                # Windows requires admin privileges for symlinks, so we'll copy instead
                import shutil
                shutil.copytree(punkt_dir, punkt_tab_dir)
                print(f"Copied punkt directory to punkt_tab to fix punkt_tab issue")
            else:  # Unix/Mac
                # Create a symlink
                os.symlink(punkt_dir, punkt_tab_dir)
                print(f"Created symlink from punkt to punkt_tab to fix punkt_tab issue")
        except Exception as e:
            print(f"Warning: Could not create punkt_tab directory: {str(e)}")
            print("Trying alternative method...")
            try:
                # If symlink fails, try copying
                import shutil
                if os.path.exists(punkt_dir):
                    # Create english directory inside punkt_tab
                    english_dir = os.path.join(punkt_tab_dir, 'english')
                    os.makedirs(english_dir, exist_ok=True)

                    # Copy PY files from punkt to punkt_tab
                    punkt_english_dir = os.path.join(punkt_dir, 'english')
                    if os.path.exists(punkt_english_dir):
                        for file in os.listdir(punkt_english_dir):
                            if file.endswith('.pickle'):
                                src = os.path.join(punkt_english_dir, file)
                                dst = os.path.join(english_dir, file)
                                shutil.copy2(src, dst)
                                print(f"Copied {file} to punkt_tab directory")

                    print("Created punkt_tab directory manually")
            except Exception as e2:
                print(f"Warning: Alternative method also failed: {str(e2)}")

    print("\nVerifying installations:")
    try:
        from nltk.corpus import stopwords
        words = stopwords.words('english')
        print(f"✓ Stopwords available ({len(words)} words)")
    except Exception as e:
        print(f"✗ Stopwords not available: {str(e)}")

    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize("This is a test sentence.")
        print(f"✓ Tokenizer available ({len(tokens)} tokens)")
    except Exception as e:
        print(f"✗ Tokenizer not available: {str(e)}")

    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize("running")
        print(f"✓ Lemmatizer available (running → {lemma})")
    except Exception as e:
        print(f"✗ Lemmatizer not available: {str(e)}")

    print("\nNLTK resources installation complete.")

if __name__ == "__main__":
    download_nltk_resources()
