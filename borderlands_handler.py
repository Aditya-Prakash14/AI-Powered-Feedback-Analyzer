import pandas as pd
import streamlit as st

def process_borderlands_format(file_path):
    """
    Special handler for the Borderlands format CSV files.
    Expected format: ID, Game, Sentiment, Text
    Example: 2401, Borderlands, Positive, im getting on borderlands and i will murder you all
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed dataframe with proper columns
    """
    try:
        # Try reading without headers first
        df = pd.read_csv(file_path, header=None)
        
        # Check if we have at least 4 columns
        if len(df.columns) >= 4:
            # Rename columns to match our expected format
            df.columns = ['id', 'platform', 'sentiment', 'text'] + [f'col_{i}' for i in range(4, len(df.columns))]
            
            # If the text column contains commas, it might have been split incorrectly
            # Let's try to fix it by combining columns after the sentiment column
            if len(df.columns) > 4:
                # Combine all columns after sentiment into a single text column
                text_columns = [f'col_{i}' for i in range(4, len(df.columns))]
                df['text'] = df['text'] + ', ' + df[text_columns].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
                df = df.drop(columns=text_columns)
            
            return df
        else:
            # If we don't have enough columns, try a different approach
            # Read the file as text and parse manually
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                # Split by comma, but only for the first 3 commas
                parts = line.split(',', 3)
                if len(parts) >= 4:
                    id_val = parts[0].strip()
                    platform = parts[1].strip()
                    sentiment = parts[2].strip()
                    text = parts[3].strip()
                    data.append([id_val, platform, sentiment, text])
                else:
                    # If we can't split properly, just use the whole line as text
                    data.append(['', 'Unknown', 'Unknown', line.strip()])
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=['id', 'platform', 'sentiment', 'text'])
            return df
    
    except Exception as e:
        st.error(f"Error processing Borderlands format: {str(e)}")
        # Fallback to a very simple approach
        try:
            # Read the file as text
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Split by newlines
            lines = content.split('\n')
            
            # Create a simple DataFrame with just the text
            df = pd.DataFrame({'text': lines})
            return df
        except:
            st.error("Failed to process the file in any format.")
            return pd.DataFrame({'text': ["Error processing file"]})

def is_borderlands_format(file_content):
    """
    Check if the file content matches the Borderlands format.
    
    Args:
        file_content: Content of the file as bytes
        
    Returns:
        bool: True if the file appears to be in Borderlands format
    """
    try:
        # Convert bytes to string
        content = file_content.decode('utf-8')
        
        # Check the first few lines
        lines = content.split('\n')[:5]  # Check first 5 lines
        
        # Count how many lines match the pattern: number, game, sentiment, text
        matches = 0
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.split(',', 3)
                if len(parts) >= 4:
                    # Check if first part is a number
                    try:
                        int(parts[0].strip())
                        # Check if third part is a sentiment
                        sentiment = parts[2].strip().lower()
                        if sentiment in ['positive', 'negative', 'neutral']:
                            matches += 1
                    except:
                        pass
        
        # If most lines match the pattern, it's likely Borderlands format
        return matches >= len([l for l in lines if l.strip()]) * 0.6
    
    except:
        return False
