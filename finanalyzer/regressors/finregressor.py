# API for the Machine Learning part of the app
#
# See the README.md file for more information.
#
# Copyright (c) 2021, Romain DODET 
# romain.dodet00@gmail.com
# All rights reserved.
#

from regressors.twitter_sentiment_analysis import TwitterSentimentAnalysis
from regressors.googletrends import GoogleTrendsAnalysis
# from NN.[regressor_name] import [regressor_name]

from typing import List
import numpy as np
import pandas as pd


class Finregressor:
    def __init__(self):
        self.twitter_sentiment_analysis = TwitterSentimentAnalysis()
        self.google_trends = GoogleTrendsAnalysis()
    
    def get_twitter_sentiment_analysis(self, tickers: List[str]) -> list:
        sentiment_list = []
        for ticker in tickers:
            sentiment_list.append(self.twitter_sentiment_analysis.get_sentiment(ticker))
        return sentiment_list
    
    def get_google_trends_analysis(self, tickers: List[str]) -> list:
        trends_list = []
        for ticker in tickers:
            trends_list.append(self.google_trends.get_interest_trend(ticker))
        return trends_list


