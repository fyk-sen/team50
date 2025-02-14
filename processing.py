"""
Processes the data.
"""
import pandas as pd

RAW = "/data/raw.csv"

data = pd.read_csv('raw.csv')
data.drop(columns=['name'], inplace=True)
