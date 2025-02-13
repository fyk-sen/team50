import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "/data/cleaned_data.csv"
MODEL_DIR = "/model_storage/"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Set up MLflow tracking
mlflow.set_tracking_uri("http://mlflow-service:5000")  # Change if MLflow is on a different service
mlflow.set_experiment("Model_Training")

def train_model():
    if not os.path.exist(DATA_PATH):
        print("No Data available")
        return
    
    # Read data
    data = pd.read_csv(DATA_PATH)
    X = data.drop("status", axis=1)
    y = data["status"]

    # Parameters
    n_estimators = 100

    # Training with Mlflow Tracking
    with mlflow.start_run():
        # Training model
        print("Training Model...")
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X, y)

    # Save model
    joblib.dumb(model, MODEL_PATH)
    print(f"Model has been saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()