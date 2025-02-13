import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "/data/cleaned_data.csv"
MODEL_PATH = "/model_storage/trained_model.pkl"

def train_model():
    if not os.path.exist(DATA_PATH):
        print("No Data available")
        return
    
    # Read data
    data = pd.read_csv(DATA_PATH)
    X = data.drop("status", axis=1)
    y = data["status"]

    # Training model
    print("Training Model...")
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Save model
    joblib.dumb(model, MODEL_PATH)
    print(f"Model has been saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()