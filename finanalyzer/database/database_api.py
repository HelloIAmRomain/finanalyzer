"""
Create an API for the finanalyzer app.
It is used if the user wants to use the API instead of the web interface.

See the README.md file for more information.
"""
import numpy as np
import pandas as pd

from database.constants import *
from database.findatabase import Findatabase

"""
class DatabaseAPI(Findatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data_from_database(self, table_name: str) -> pd.DataFrame:
        """
        Get data from the database.

        Parameters
        ----------
        table_name : str
            The name of the table in the database.

        Returns
        -------
        pd.DataFrame
            The data from the database.
        """
        return self.read_data_from_database(table_name)
"""