"""
Processes the data.
"""
import pandas as pd
from sklearn.impute import KNNImputer

RAW = "/data/raw.csv"

data = pd.read_csv('raw.csv')

df = data
df.drop(columns=['name'], inplace=True)
df.drop_duplicates(inplace=True)

imputer = KNNImputer(n_neighbors=1)
df[:] = imputer.fit_transform(df)

df['status'] = data['status'].astype('bool')
