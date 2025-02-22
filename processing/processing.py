"""
Processes the data.
"""
import pandas as pd
from flask import Flask
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
import os

RAW_TRAIN = "/data/raw_train.csv"
RAW_TEST = "/data/raw_test.csv"

PROCESSED_TEST = "/data/processed_test.csv"

X_TRAIN = "/data/x_train.csv"
Y_TRAIN = "/data/y_train.csv"
X_TEST = "/data/x_test.csv"
Y_TEST = "/data/y_test.csv"


def clean(clean_df):
    """
    Cleans the data.
    """

    if 'name' in clean_df.columns:
        clean_df.drop(columns=['name'], inplace=True)

    clean_df.drop_duplicates(inplace=True)

    imputer = KNNImputer(n_neighbors=1)
    clean_df.loc[:,:] = imputer.fit_transform(clean_df)

    return clean_df


def split(split_df):
    """
    Splits the data.
    """
    split_df.status = split_df.status.astype('bool')

    feature_cols = [x for x in split_df.columns if x != 'status']

    strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                               random_state=42)

    train_idx, test_idx = next(strat_shuff_split.split(split_df[feature_cols],
                                                       split_df.status))

    x_train = split_df.loc[train_idx, feature_cols]
    y_train = split_df.loc[train_idx, 'status']

    x_test = split_df.loc[test_idx, feature_cols]
    y_test = split_df.loc[test_idx, 'status']

    x_train.to_csv(X_TRAIN, index=False)
    y_train.to_csv(Y_TRAIN, index=False)

    x_test.to_csv(X_TEST, index=False)
    y_test.to_csv(Y_TEST, index=False)


app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    """
    Processes the data (actually).
    """
    if os.path.exists(RAW_TRAIN):
        raw_train = pd.read_csv(RAW_TRAIN)
        df = raw_train
        clean(df)
        split(df)
        return "train processing completed", 200

    elif os.path.exists(RAW_TEST):
        raw_test = pd.read_csv(RAW_TEST)
        df = clean(raw_test)  # Ensure cleaned data is saved
        df.to_csv(PROCESSED_TEST, index=False)

        if os.path.exists(PROCESSED_TEST):
            print(f"✅ Processed test data saved at {PROCESSED_TEST}")
        else:
            print("❌ Error: Processed test data not found after saving.")

        return "test processing completed", 200

    else:
        return "no data file found", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
