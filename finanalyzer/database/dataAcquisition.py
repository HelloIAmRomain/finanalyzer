import time
from typing import Tuple
import pandas as pd
import yfinance as yf
from fp.fp import FreeProxy
from constants import TICKERS_CSV, WAIT_TIME_BETWEEN_REQUESTS

"""
------------------------
msft = yf.Ticker("MSFT")
msft.info
msft.history(period)
------------------------
data = yf.download("SPY AAPL MSFT TWTR", start="2022-01-01", end="2022-05-30")
data['SPY']['Close']
"""


# For future: check proxy connexion
def get_proxy() -> FreeProxy:
    proxy = FreeProxy(rand=True, anonym=True).get()
    print(proxy)
    return proxy


def get_tickers(TICKERS_CSV=TICKERS_CSV) -> Tuple[Tuple]:
    companies_csv = pd.read_csv(TICKERS_CSV, sep=",")
    stock_tickers = tuple(companies_csv["Ticker"])
    companies_names = tuple(companies_csv["Name"])
    exchange_platform = tuple(companies_csv["Exchange"])
    if WAIT_TIME_BETWEEN_REQUESTS:
        time.sleep(WAIT_TIME_BETWEEN_REQUESTS)
    return stock_tickers, companies_names, exchange_platform


def receive_history(ticker: str,
                    columns: list = ["Open"],
                    interval: str = "1d",
                    period: str = None,
                    start=None,
                    proxy=None,
                    ) -> tuple:
    """ Receive history data from yfinance."""
    data = yf.Ticker(ticker)
    if WAIT_TIME_BETWEEN_REQUESTS:
        time.sleep(WAIT_TIME_BETWEEN_REQUESTS)
    if start is not None:
        financials = data.history(interval=interval, start=start, proxy=proxy)
    else:
        financials = data.history(interval=interval, period=period, proxy=proxy)
    return financials[list(columns)]


def receive_info(ticker: str,
                 columns: list,
                 proxy=None
                 ) -> list:
    data = yf.Ticker(ticker).info
    # 0 si la donnee n'existe pas (None), ou n'est pas recue (column pas dans data)
    list_info = []
    for col in columns:
        if col not in data.keys():
            data[col] = 0
        if data[col] is None:
            data[col] = 0
        list_info.append(data[col])
    return list_info


def check_exists(ticker: str, period="1m", proxy=None, start=None) -> bool:
    exists = True
    if start:
        val = receive_history(ticker=ticker, start=start, proxy=proxy)
    else:
        val = receive_history(ticker, period=period, proxy=proxy)
    if len(val) == 0:
        exists = False
    return exists


"""
TODO:
--> diminuer le nombre de requetes en les regoupant
def recive_multiple_data(tickers:str, columns:list, period="2y", interval:str ="1d", start=None, proxy=None):
    data = yf.Tickers(tickers)
    if start is not None:
        financials = data.history(interval=interval, start=start, proxy=proxy)
    else:
        financials = data.history(interval=interval, period=period, proxy=proxy)
    for companies in tickers.split():
"""

"""
------------------- AUTRES MODULES possibles ----------------------------
investpy
nasdaqdatalink
wallstreet
pandas-datareader
"""
