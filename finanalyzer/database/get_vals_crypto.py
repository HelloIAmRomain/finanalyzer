# Script that gets the n last values of a given cryptocurrency

import sys
import requests
import json
import time
import datetime
import os
import pandas as pd


def get_vals(coin_name: str, n: int, interval: str, aggregate: int, exchange: str='CCCAGG'):
    """
    Gets the n last values of a given cryptocurrency
    :param coin_name: Name of the cryptocurrency
    :param n: Number of values to get
    :param interval: Interval of the values to get
    :return: List of the n last values
    """
    # Get the values (every day)
    # url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={coin_name}&tsym=USD&limit={n}&aggregate=1&e=CCCAGG'
    parameters = {
        'fsym': coin_name,
        'tsym': 'USD',
        'e': exchange,
        'aggregate': aggregate,
        'limit': n,
    }
    api_key = "dcbfffa9b25a1bab3ab3e662c265b99fb3c4452fcfb75f896d1b73a7c90bdef4"
    url = f'https://min-api.cryptocompare.com/data/v2/histominute?{"&".join([f"{k}={v}" for k, v in parameters.items()])}&api_key={api_key}'





    response = requests.get(url)
    data = json.loads(response.text)
    if data['Response'] != 'Success':
        raise Exception('Error getting the values')

    time_from = data['Data']['TimeFrom']
    time_to = data['Data']['TimeTo']
    values = data['Data']['Data']

    # Get the timestamps
    timestamps = []
    for value in values:
        timestamps.append(value['time'])

    # Get the values
    values = []
    for value in values:
        values.append(value['close'])

    return values, timestamps