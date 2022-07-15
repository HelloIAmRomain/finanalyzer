import sqlite3
from sqlite3 import Error
import yfinance as yf
from typing import Tuple
import pandas as pd
from datetime import datetime, timedelta
import constants
import dataAcquisition as da
import sys
import os


class Findatabase:
    """
    Permet d'accéder à la base de données financière
    Schema de la BDD (A mettre à jour en cas de changement):

           identification: TABLE_NAMES
        -------------------------------
        | ID | Nom | Ticker | Echange |
        -------------------------------

           data: TABLE_VALUES
        ----------------------------------
        | ID | Open | High | Low | Close |
        ----------------------------------

        Table info (voir le fichier SQL)

Attention, pour le moment le proxy n'est pas encore utilisé pour la table Infos
    """

    def __init__(self,
                 db_file=constants.DEFAULT_DB_FILE,
                 reinitialize: bool = False,
                 proxy=True,
                 MAX_DATA_HISTORY_YEARS=constants.MAX_DATA_HISTORY_YEARS,
                 MAX_REQUESTS_BEFORE_PROXY_CHANGE=constants.MAX_REQUESTS_BEFORE_PROXY_CHANGE
                 ):
        self.TABLE_NAMES = constants.TABLE_NAMES
        self.TABLE_VALUES = constants.TABLE_VALUES
        self.TABLE_INFO = constants.TABLE_FINANCIAL_DATA
        self.number_req_proxy = [0, MAX_REQUESTS_BEFORE_PROXY_CHANGE]
        self.proxy = da.get_proxy() if proxy else None
        self.MAX_DATA_HISTORY_YEARS = MAX_DATA_HISTORY_YEARS
        self.db_file = db_file
        self.con = sqlite3.connect(self.db_file)
        self.today = self.get_date()
        self.yesterday = self.get_date(delta_days=1)
        self.start = self.get_date(delta_days=self.MAX_DATA_HISTORY_YEARS * 365)
        if reinitialize:
            self.initialize_database()
        self.COLUMNS_INFO = self.get_column_names(self.TABLE_INFO)[
                            :-2]  # Ne pas mettre les dernieres colonnes (date d'ajout et id entreprise)
        self.COLUMNS_VALUES = self.get_column_names(self.TABLE_VALUES)
        self.COLUMNS_NAMES = self.get_column_names(self.TABLE_NAMES)

    def __str__(self):
        info = f"""Merci d'utiliser Findatabase
Cette classe permet d'accéder à la  base de données {self.db_file}.
Les tables sont {self.TABLE_NAMES} et {self.TABLE_VALUES}.
Les données sont rafraichies chaque jour.
Les données utilisées sont l'ensemble des jours ouvrés entre aijourd'hui et il y a {self.MAX_DATA_HISTORY_YEARS} ans, soit depuis le {self.start}.
Lorsque vous avez terminé, vous pouvez fermer manuellement la base de données avec foo.close()
"""
        return info

    def check_last_update(self, id_ticker: int, table: str):
        """
        Retourne la date de la dernière mise à jour de la table des données
        """
        data = self.read_database(table=table, element="dateValue",
                                  optional=f"WHERE namesId={id_ticker} ORDER BY dateValue DESC")
        if data:
            return data[0][0];
        else:
            return None

    def initialize_database(self, fill_data=False):
        with open(self.db_file, 'w'):
            pass
        # instruction ecrit dans un fichier pour plus de clarté
        with open(constants.SQL_INITIALIZE_DATABASE, "r") as f:
            instructions = f.read()
        self.write_database(script=instructions)
        self.delete_values_table(self.TABLE_NAMES)
        self.delete_values_table(self.TABLE_VALUES)
        self.delete_values_table(self.TABLE_INFO)
        self.set_database_names()
        if fill_data:
            self.update_database()

    def write_database(self, sql_instruction=None, script=None, n_requests=0):
        feedback = None
        # if n_reques
        try:
            cur = self.con.cursor()
            if script is None:
                cmd = cur.execute(sql_instruction)
            else:
                cmd = cur.executescript(script)
            feedback = tuple(cmd)
            self.con.commit()
        except Error as e:
            print(e)
        return feedback

    def get_date(self, delta_days: int = 0) -> str:
        date = datetime.strftime(datetime.now() - timedelta(delta_days), format='%Y-%m-%d')
        return date

    def get_ticker_from_id(self, id_ticker: int) -> str:
        data = self.read_database(table=self.TABLE_NAMES, element="ticker", optional=f"WHERE id={id_ticker}")
        if data:
            return data[0][0]
        else:
            return None

    def get_column_names(self, table: str) -> tuple:
        data = self.write_database(sql_instruction=f"PRAGMA table_info({table});")
        columns_names = tuple(d[1] for d in data)
        return columns_names

    def read_database(self, table: int, element="*", optional="") -> Tuple[Tuple]:
        data = self.write_database(sql_instruction=f"SELECT {element} FROM {table} {optional};")
        return data

    def request_count(self):
        self.number_req_proxy[0] += 1
        if self.number_req_proxy[0] == self.number_req_proxy[1] and self.proxy is not None:
            self.proxy = da.get_proxy()
            self.number_req_proxy[0] = 0
            print(f"Proxy changed to {self.proxy}")

    def add_data(self, data: tuple or list):
        instr = ",".join("'" + i + "'" if isinstance(i, str) else str(i) for i in data)
        self.write_database(sql_instruction=f'INSERT INTO {self.TABLE_VALUES} VALUES ({instr});')

    def add_name(self, data: tuple or list):
        instr = ",".join("'" + i + "'" if isinstance(i, str) else str(i) for i in data)
        self.write_database(sql_instruction=f'INSERT INTO {self.TABLE_NAMES} VALUES ({instr});')

    def add_info(self, data: tuple or list):
        instr = ",".join("'" + i + "'" if isinstance(i, str) else str(i) for i in data)
        self.write_database(sql_instruction=f'INSERT INTO {self.TABLE_INFO} VALUES ({instr});')

    def set_database_names(self):
        tickers, companies_names, exchange = da.get_tickers()
        for n in range(len(tickers)):
            if self.read_database(table=self.TABLE_NAMES,
                                  optional=f"WHERE ticker=\"{tickers[n]}\""):
                # verifie si le ticker n'est pas deja présent dans la base de donnée
                print(f"Numero {n}: {tickers[n]} est deja dans la base de donnees")
            else:
                self.request_count()
                if da.check_exists(tickers[n], proxy=self.proxy):
                    print("Numero: ", n, "ticker: ", tickers[n], "added")
                    entry = (n + 1, tickers[n], companies_names[n], exchange[n])
                    self.add_name(entry)
                else:
                    print(f"Numero {n}: {tickers[n]} n\'existe pas")

    def insert_history_from_web(self, id_name: int, start: str):
        self.request_count()
        ticker = self.get_ticker_from_id(id_name)
        values = da.receive_history(ticker, start=start, columns=constants.DEFAULT_COLUMNS_HISTORY, proxy=self.proxy)
        dates_value = list(values.index)
        data = values.to_numpy()
        for val, date_value in zip(data, dates_value):
            # Pas optimal: tate_value = "yyyy-mm-dd hh:mm:ss", on recupere donc les 10 premiers, soit "yyyy-mm-dd"
            line = list(val) + [str(date_value)[:10], self.today, id_name]
            self.add_data(data=line)

    def insert_info_from_web(self, id_name: int):
        self.request_count()
        ticker = self.get_ticker_from_id(id_name)
        info = da.receive_info(ticker, columns=self.COLUMNS_INFO, proxy=self.proxy)
        info_line = list(info) + [self.today, id_name]
        self.add_info(data=info_line)

    def fill_all_data(self, history=True, info=True):
        """
        Remplit toutes les valeurs de toutes les entreprises
        Si le tableau de valeurs contient des éléments, ils seront supprimés
        """
        if history:
            self.delete_values_table(self.TABLE_VALUES)
        if info:
            self.delete_values_table(self.TABLE_INFO)
        tickers = self.read_database(table=self.TABLE_NAMES,
                                     element="ticker")  # pb: on a un resultat de la forme ((a),(b),...)
        tickers = (t[0] for t in tickers)
        id_names = self.read_database(table=self.TABLE_NAMES, element="id")  # meme probleme
        id_names = (i[0] for i in id_names)
        for id_name in id_names:
            if history:
                self.insert_history_from_web(id_name=id_name, start=self.start)
            if info:
                self.insert_info_from_web(id_name=id_name)
        self.delete_redundancy()
        print("All done: updated")

    def update_database(self):
        """
        Met à jour la base de données
        > Si il n'y a pas de données, on les crée
        > Si il y a des données, on les met à jour et on supprime les anciennes
        """
        for id_name in self.read_database(table=self.TABLE_NAMES, element="id"):
            id_name = id_name[0]
            last_update_info = self.check_last_update(id_ticker=id_name, table=self.TABLE_INFO)
            last_update_history = self.check_last_update(id_ticker=id_name, table=self.TABLE_VALUES)
            print(f"----- ID = {id_name} - Ticker = {self.get_ticker_from_id(id_name)} -----")
            if last_update_history is None:
                self.insert_history_from_web(id_name=id_name, start=self.start)  # No values - get all values
                print(": There was no nata before... Updated")
            elif last_update_history != self.yesterday and last_update_history != self.today:
                next_day = datetime.strftime(datetime.strptime(last_update_history, "%Y-%m-%d") + timedelta(days=1),
                                             "%Y-%m-%d")
                self.insert_history_from_web(id_name=id_name, start=next_day)
                print(
                        f"historical data was not up to date: last recorded date: {last_update_history}... Updated from {next_day}")
            else:
                print('historical data already up-to-date')
            if last_update_info != self.today:
                print("info... Updated")
                self.insert_info_from_web(id_name=id_name)
            else:
                print("info already already up-to-date")
        self.delete_values_table(self.TABLE_VALUES, optional=f"WHERE dateAdded < '{self.start}'")
        self.delete_redundancy()
        print("All done: updated")

    def delete_values_table(self, table: str, optional=""):
        self.write_database(script=f'DELETE FROM {table} {optional};')

    def close_database(self):
        self.con.close()

    def delete_redundancy(self):
        with open(constants.SQL_REMOVE_REDUNDANCY, "r") as f:
            instructions = f.read()
        self.write_database(script=instructions)


if __name__ == "__main__":
    db = Findatabase(reinitialize=False, proxy=True)
    db.update_database()
