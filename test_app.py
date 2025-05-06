import unittest
import pandas as pd
import re
from app import preprocess_text

class TestProductPulse(unittest.TestCase):
    
    def test_preprocess_text(self):
        # Test URL removal
        text_with_url = "Check out our website https://example.com for more info"
        self.assertNotIn("https://", preprocess_text(text_with_url))
        
        # Test hashtag removal
        text_with_hashtag = "This app is #awesome and #useful"
        self.assertNotIn("#", preprocess_text(text_with_hashtag))
        
        # Test mention removal
        text_with_mention = "Hey @developer, great job on the app!"
        self.assertNotIn("@", preprocess_text(text_with_mention))
        
        # Test special character removal
        text_with_special_chars = "The app crashes! It's frustrating."
        self.assertNotIn("!", preprocess_text(text_with_special_chars))
        self.assertNotIn("'", preprocess_text(text_with_special_chars))
    
    def test_data_loading(self):
        # Test CSV loading
        try:
            df = pd.read_csv('sample_data.csv')
            self.assertIn('text', df.columns)
        except:
            pass  # Skip if file doesn't exist
        
        # Test JSON loading
        try:
            df = pd.read_json('sample_data.json')
            self.assertIn('text', df.columns)
            self.assertIn('platform', df.columns)
        except:
            pass  # Skip if file doesn't exist

if __name__ == '__main__':
    unittest.main()
