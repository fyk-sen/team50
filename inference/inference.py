import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

MODEL_PATH = "/model-storage/trained_model.pkl"
METRICS_PATH = "/model-storage/metrics.csv"  # File to save metrics
X_TEST_PATH = "./data/x_test.csv"
Y_TEST_PATH = "./data/y_test.csv"

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return

    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        print("Test data files not found. Ensure test data and labels are available.")
        return

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    Y_test = pd.read_csv(Y_TEST_PATH).values.ravel()  # Flatten if necessary

    # Ensure features are correctly formatted for prediction
    Y_pred = model.predict(X_test)

    # Prepare metrics output
    metrics_output = ["Model Evaluation Metrics:\n"]

    if hasattr(model, "predict_proba"):  # Classification model
        accuracy = accuracy_score(Y_test, Y_pred)
        cm = confusion_matrix(Y_test, Y_pred)

        metrics_output.append(f"Accuracy: {accuracy:.4f}\n")
        metrics_output.append("Confusion Matrix:\n")
        metrics_output.append(np.array2string(cm) + "\n")

    else:  # Regression model
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        metrics_output.append(f"Mean Squared Error: {mse:.4f}\n")
        metrics_output.append(f"R² Score: {r2:.4f}\n")

    # Save metrics to PVC
    with open(METRICS_PATH, "w") as f:
        f.writelines(metrics_output)

    print("Metrics saved to PVC at:", METRICS_PATH)

if __name__ == "__main__":
    evaluate_model()
