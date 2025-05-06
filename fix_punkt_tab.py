#!/usr/bin/env python3
"""
This script fixes the punkt_tab issue in NLTK by creating a symlink or copy
from the punkt directory to the punkt_tab directory.
"""

import os
import sys
import shutil

def fix_punkt_tab_issue():
    """
    Fix the punkt_tab issue by creating a symlink or copy from punkt to punkt_tab.
    """
    # Get NLTK data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    
    # Check if NLTK data directory exists
    if not os.path.exists(nltk_data_dir):
        print(f"NLTK data directory not found: {nltk_data_dir}")
        print("Please run 'python -m nltk.downloader punkt' first.")
        return False
    
    # Define paths
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab')
    
    # Check if punkt directory exists
    if not os.path.exists(punkt_dir):
        print(f"Punkt directory not found: {punkt_dir}")
        print("Please run 'python -m nltk.downloader punkt' first.")
        return False
    
    # Check if punkt_tab directory already exists
    if os.path.exists(punkt_tab_dir):
        print(f"Punkt_tab directory already exists: {punkt_tab_dir}")
        return True
    
    # Create punkt_tab directory
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(punkt_tab_dir), exist_ok=True)
        
        # Create a symlink or copy from punkt to punkt_tab
        if os.name == 'nt':  # Windows
            # Windows requires admin privileges for symlinks, so we'll copy instead
            shutil.copytree(punkt_dir, punkt_tab_dir)
            print(f"Copied punkt directory to punkt_tab: {punkt_tab_dir}")
        else:  # Unix/Mac
            # Create a symlink
            os.symlink(punkt_dir, punkt_tab_dir)
            print(f"Created symlink from punkt to punkt_tab: {punkt_tab_dir}")
        
        return True
    
    except Exception as e:
        print(f"Error creating punkt_tab directory: {str(e)}")
        print("Trying alternative method...")
        
        try:
            # If symlink fails, try copying
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
                            print(f"Copied {file} to punkt_tab directory")
                
                print(f"Created punkt_tab directory manually: {punkt_tab_dir}")
                return True
            else:
                print(f"Punkt directory not found: {punkt_dir}")
                return False
        
        except Exception as e2:
            print(f"Alternative method also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    print("Fixing punkt_tab issue in NLTK...")
    if fix_punkt_tab_issue():
        print("Successfully fixed punkt_tab issue.")
        sys.exit(0)
    else:
        print("Failed to fix punkt_tab issue.")
        sys.exit(1)
