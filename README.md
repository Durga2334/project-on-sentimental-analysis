import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the required resources for the first time
nltk.download('vader_lexicon')

# Sample text for sentiment analysis
sample_text = "I love how this product works! It's amazing."

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the sample text
sentiment_scores = sia.polarity_scores(sample_text)

# Determine the overall sentiment label
if sentiment_scores['compound'] >= 0.05:
    sentiment_label = 'Positive'
elif sentiment_scores['compound'] <= -0.05:
    sentiment_label = 'Negative'
else:
    sentiment_label = 'Neutral'

# Print the sentiment score and label
print("Sample Text:", sample_text)
print("Sentiment Scores:", sentiment_scores)
print("Sentiment Label:", sentiment_label)

# Another sample text
sample_text_2 = "This movie is so boring and disappointing."

# Perform sentiment analysis on the second sample text
sentiment_scores_2 = sia.polarity_scores(sample_text_2)

# Determine the overall sentiment label for the second text
if sentiment_scores_2['compound'] >= 0.05:
    sentiment_label_2 = 'Positive'
elif sentiment_scores_2['compound'] <= -0.05:
    sentiment_label_2 = 'Negative'
else:
    sentiment_label_2 = 'Neutral'

# Print the sentiment score and label for the second text
print("Sample Text 2:", sample_text_2)
print("Sentiment Scores 2:", sentiment_scores_2)
print("Sentiment Label 2:", sentiment_label_2)

# More advanced example using a list of sample texts
sample_texts = [
    "This book is fantastic!",
    "The weather is dreadful today.",
    "I'm feeling great about the upcoming event.",
    "This restaurant serves amazing food, I highly recommend it.",
    "I can't believe how bad the service is here.",
    "The new update of the app is quite buggy.",
]

# Perform sentiment analysis on the list of sample texts
for i, text in enumerate(sample_texts):
    sentiment_scores_i = sia.polarity_scores(text)
    if sentiment_scores_i['compound'] >= 0.05:
        sentiment_label_i = 'Positive'
    elif sentiment_scores_i['compound'] <= -0.05:
        sentiment_label_i = 'Negative'
    else:
        sentiment_label_i = 'Neutral'
    print(f"Sample Text {i+1}: {text}")
    print(f"Sentiment Scores {i+1}: {sentiment_scores_i}")
    print(f"Sentiment Label {i+1}: {sentiment_label_i}")
    print()
