import pandas as pd
import numpy as np
import datetime

# Create sample data
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

# Create DataFrame with different column names to test column mapping
sample_df = pd.DataFrame({
    'feedback_text': np.random.choice(texts, size=30, replace=True),
    'source': np.random.choice(platforms, size=30, replace=True),
    'feedback_date': dates,
    'user_id': [f"user_{i}" for i in range(1, 31)],
    'rating': np.random.randint(1, 6, size=30)
})

# Save to Excel
sample_df.to_excel('sample_data.xlsx', index=False)
print("Excel sample file created: sample_data.xlsx")

# Also create a CSV with different delimiter and encoding
sample_df.to_csv('sample_data_semicolon.csv', sep=';', index=False)
print("CSV sample file with semicolon delimiter created: sample_data_semicolon.csv")

# Create a CSV with different column names
sample_df.rename(columns={
    'feedback_text': 'comment',
    'source': 'channel',
    'feedback_date': 'timestamp'
}).to_csv('sample_data_different_columns.csv', index=False)
print("CSV sample file with different column names created: sample_data_different_columns.csv")
