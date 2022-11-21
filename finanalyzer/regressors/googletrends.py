# Google Trends analysis for the selected companies
#
# See the README.md file for more information.
#
# Copyright (c) 2021, Romain DODET 
# romain.dodet00@gmail.com
# All rights reserved.
#

import pandas as pd                        
from pytrends.request import TrendReq
import numpy as np
import requests
import json
import datetime
import time
import matplotlib.pyplot as plt
import os
import sys


class GoogleTrendsAnalysis:
    def __init__(self):
        self.trend = TrendReq()

    def get_trend(self, company:str, date_start:str, date_end:str) -> pd.DataFrame:
        self.trend.build_payload(kw_list=[company], timeframe=date_start + ' ' + date_end)
        # Interest at the date
        interest = self.trend.interest_over_time()
        interest = interest[company].to_numpy()
        return interest
    
    def get_interest_trend(self, company:str, date:str, period_comparison:int=365) -> float:
        # Get the first date of the period
        date_end = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_start = date_end - datetime.timedelta(days=period_comparison)
        date_start = date_start.strftime("%Y-%m-%d")
        date_end = date
        # Get the interest at the period
        interest = self.get_trend(company, date_start=date_start, date_end=date_end)
        # Get the slope of the trend
        slope = np.polyfit(np.arange(len(interest)), interest, 1)[0]
        return slope
        
    

if __name__ == "__main__":
    analytics = GoogleTrendsAnalysis()
    val = analytics.get_trend('Apple', '2021-01-01', '2021-01-31')
    plt.plot(val)
    plt.title('Google Trends for Apple', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Interest', fontsize=14)
    plt.show()
    
    interest = analytics.get_interest_trend('Apple', '2021-01-31', period_comparison=365)
    print("Interest rate evolution for Apple the past year: ", interest)
    
