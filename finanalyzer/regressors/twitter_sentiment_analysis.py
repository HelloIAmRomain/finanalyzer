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

# English stopwords
stop_words = set(stopwords.words('english'))


class TwitterSentimentAnalysis:
    """
    Text analysis from tweets available on Twitter about a company
    Output: a sentiment score
    """
    def __init__(self, consumer_key: str, consumer_secret: str, access_token: str, access_token_secret: str):
        self.TWITTER_TOKEN = ""