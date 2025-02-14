"""
Processes the data.
"""
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit

RAW = "/data/raw.csv"

data = pd.read_csv('raw.csv')
df = data

df.drop(columns=['name'], inplace=True)
df.drop_duplicates(inplace=True)

imputer = KNNImputer(n_neighbors=1)
df[:] = imputer.fit_transform(df)

df['status'] = data['status'].astype('bool')

feature_cols = [x for x in df.columns if x != 'status']

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                           random_state=42)

train_idx, test_idx = next(strat_shuff_split.split(df[feature_cols], 
                                                   df['status']))
X_train = df.loc[train_idx, feature_cols]
y_train = df.loc[train_idx, 'status']

X_test = df.loc[test_idx, feature_cols]
y_test = df.loc[test_idx, 'status']
