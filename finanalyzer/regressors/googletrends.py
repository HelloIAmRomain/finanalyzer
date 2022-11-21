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
import os
import sys


class GoogleTrendsAnalysis:
    def __init__(self):
        self.trend = TrendReq()

    def get_trend(self, company:str, date:str) -> float:
        self.trend.build_payload(kw_list=[company], timeframe=date)
        # Interest at the date
        interest = self.trend.interest_over_time()[company][0]
        return interest
