# Create an API for the finanalyzer app.
# It is used if the user wants to use the API instead of the web interface.

# See the README.md file for more information.
#
# Copyright (c) 2021, Romain DODET 
# romain.dodet00@gmail.com
# All rights reserved.
#

from database.constants import *
from database.findatabase import Findatabase
from regressors.finregressor import Finregressor

from typing import List
import numpy as np
import pandas as pd
import streamlit as st




class FinanalyzerAPI:
    def __init__(self, database: Findatabase, regressor: Finregressor):
        self.database = database
        self.regressor = regressor
        self.frontend = init_frontend()
    
    def init_frontend(self):
        st.set_page_config(page_title="Finanalyzer", page_icon=":chart_with_upwards_trend:", layout="wide")
        st.title("📈 Finanalyzer")
        st.subheader("Financial data analysis and prediction")
        
        st.sidebar.title("📊 Data selection")
        st.sidebar.subheader("Select the data you want to analyze")
        list_tickers = self.database.list_tickers()
        list_tickers_selected = st.sidebar.multiselect("Tickers", list_tickers)
        start_date = st.sidebar.date_input("🗓️ Start date")
        # Si l'utilisateur ne sélectionne pas de date de début, ou si la date de début est postérieure à la date actuelle, la date de début est fixée à la première date de la base de données.
        # Egalement, si la date de début est antérieure à la première date de la base de données, la date de début est fixée à la première date de la base de données.
        
        st.sidebar.subheader("Select the data you want to predict")
        # Plot the data
        st.sidebar.subheader("Plot the data")
        plot_checked = st.sidebar.checkbox("Plot the data")
        if plot_checked:
            list_id_selected = self.database.get_id_from_tickers(list_tickers_selected)
            list_vals = pd.DataFrame()
            for id in list_id_selected:
                list_vals[id] = self.database.read_values_from_id(id, element="Close", start_date=start_date)
            st.line_chart(list_vals)