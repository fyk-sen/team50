import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "/mnt/data/cleaned_data.csv"
MODEL_DIR = "/mnt/model_storage/"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Set up MLflow tracking
mlflow.set_tracking_uri("http://mlflow-service:5000")  # Change if MLflow is on a different service
mlflow.set_experiment("Model_Training")

def train_model():
    """
    function to train model
    """
    if not os.path.exists(DATA_PATH):
        print("No Data available")
        return
    
    # Read data
    data = pd.read_csv(DATA_PATH)
    x = data.drop("status", axis=1)
    y = data["status"]

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in stratified_split.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Parameters
    n_estimators = 100

    # Training with Mlflow Tracking
    with mlflow.start_run():

        # Training model
        print("Training Model...")
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(x_train, y_train)

        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Compute Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Log parameters & metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_r2_score", train_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2_score", test_r2)

        # Log model in MLflow
        mlflow.sklearn.log_model(model, "trained_rf_model")
        print("Model logged to MLflow")

        # Save model
        joblib.dump(model, MODEL_PATH)
        print(f"Model has been saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
