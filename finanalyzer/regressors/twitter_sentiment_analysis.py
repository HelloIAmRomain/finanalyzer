# Twitter sentiment analysis for the selected tickers (takes a name of a company and returns a sentiment score)
#
# See the README.md file for more information.
#
# Copyright (c) 2021, Romain DODET 
# romain.dodet00@gmail.com
# All rights reserved.
#

import pandas as pd
import numpy as np
import tweepy as tw
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
# from googletrans import Translator

# English stopwords
stop_words = set(stopwords.words('english'))
# Lemmatizer
lemmatizer = WordNetLemmatizer()
# Translator
## translator = Translator()

# Twitter API credentials
TWITTER_API_KEY = 'TWITTER_API_KEY'
TWITTER_API_KEY_SECRET = 'TWITTER_API_KEY_SECRET'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'
# Access Token and Access Token Secret (with read-only access)
TWITTER_ACCESS_TOKEN = 'TWITTER_ACCESS_TOKEN'
TWITTER_ACCESS_TOKEN_SECRET = 'TWITTER_ACCESS_TOKEN_SECRET'



def clean_text(text):
    # Remove accent by converting to ASCII
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove @mentions
    text = re.sub(r'#', '', text) # remove # symbol
    text = re.sub(r'RT[\s]+', '', text) # remove RT
    text = re.sub(r'https?:\/\/\S+', '', text) # remove hyperlink
    text = re.sub(r'[^a-zA-Z]', ' ', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = text.strip() # remove leading and trailing spaces
    text = re.sub(r'\b\w{1,2}\b', '', text) # remove words with 1 or 2 letters
    text = re.sub(r'\b\w{20,}\b', '', text) # remove words with more than 20 letters
    text = re.sub(u"[àáâãäå]", 'a', text)
    text = re.sub(u"[èéêë]", 'e', text)
    text = re.sub(u"[ìíîï]", 'i', text)
    text = re.sub(u"[òóôõö]", 'o', text)
    text = re.sub(u"[ùúûü]", 'u', text)
    text = re.sub(u"[ýÿ]", 'y', text)
    text = re.sub(u"[ß]", 'ss', text)
    text = re.sub(u"[ñ]", 'n', text)
    return text

def translate_to_english(text):
    # Get language of the text
    language = detect(text)
    if language != 'en':
        # TODO: If the language is not English, translate it
        text = ""
    return text



def give_sentiment_score(text_list):
    # Replace accented characters with unaccented characters
    text_list = [clean_text(text) for text in text_list]
    # Remove empty strings
    text_list = [text for text in text_list if text != '']
    text_list = [translate_to_english(text) for text in text_list]
    # Remove empty strings (TODO: remove this line when the translation is implemented)
    text_list = [text for text in text_list if text != '']
    # Tokenize
    text_list = [word_tokenize(text) for text in text_list]
    # Remove stopwords
    text_list = [[word for word in text if word not in stop_words] for text in text_list]
    # Lemmatize
    text_list = [[lemmatizer.lemmatize(word) for word in text] for text in text_list]
    # Join the tokens
    text_list = [' '.join(text) for text in text_list]
    # Sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = [sid.polarity_scores(text) for text in text_list]
    # Get the compound score
    sentiment_scores = [score['compound'] for score in sentiment_scores]
    # Get the mean of the compound scores
    sentiment_score = np.mean(sentiment_scores)
    return sentiment_score

class TwitterSentimentAnalysis:
    """
    Text analysis from tweets available on Twitter about a company
    Output: a sentiment score
    """
    def __init__(self):
        self.api = self.connect_to_twitter()

    def connect_to_twitter(self):
        # Connect to Twitter API
        client = tw.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_KEY_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
        )
        return client

    def get_tweets(self, company, max_tweets=1000):
        # Get the tweets about the company
        tweets = self.api.search_recent_tweets(
            query=company,
            max_results=max_tweets
        )
        tweets_list = [tweet.text for tweet in tweets.data]
        return tweets_list

    def get_sentiment_score(self, tweets):
        # Get the sentiment score
        sentiment_score = 0
        if len(tweets) > 0:
            sentiment_score = give_sentiment_score(tweets)
        return sentiment_score

    def get_score(self, company, max_tweets=100):
        # Get the score for the company
        tweets = self.get_tweets(company, max_tweets)
        sentiment_score = self.get_sentiment_score(tweets)
        return sentiment_score


if __name__ == "__main__":
    # Example
    company = "AAPL"
    max_tweets = 100
    # RFC3339 format (YYYY-MM-DDTHH:MM:SSZ)
    # date_since = "2021-01-01T00:00:00Z"
    # date_until = "2021-01-31T00:00:00Z"
    twitter_sentiment_analysis = TwitterSentimentAnalysis()
    sentiment_score = twitter_sentiment_analysis.get_score(company, max_tweets)
    print(sentiment_score)
