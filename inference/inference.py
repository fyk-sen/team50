from flask import Flask, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# File paths
MODEL_PATH = "/model-storage/trained_model.pkl"
METRICS_PATH = "/model-storage/metrics.csv"
X_TEST_PATH = "./data/x_test.csv"
Y_TEST_PATH = "./data/y_test.csv"

def evaluate_model():
    """Runs model evaluation and saves metrics."""
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found. Train the model first."}, 404

    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        return {"error": "Test data files not found. Ensure test data and labels are available."}, 404

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    Y_test = pd.read_csv(Y_TEST_PATH).values.ravel()  # Flatten if necessary

    # Predict
    Y_pred = model.predict(X_test)

    # Prepare metrics
    metrics_data = {}

    if hasattr(model, "predict_proba"):  # Classification model
        accuracy = accuracy_score(Y_test, Y_pred)
        cm = confusion_matrix(Y_test, Y_pred).tolist()  # Convert to list for JSON serialization

        metrics_data["accuracy"] = round(accuracy, 4)
        metrics_data["confusion_matrix"] = cm

    else:  # Regression model
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        metrics_data["mean_squared_error"] = round(mse, 4)
        metrics_data["r2_score"] = round(r2, 4)

    # Save to PVC (CSV format)
    df_metrics = pd.DataFrame.from_dict(metrics_data, orient="index", columns=["Value"])
    df_metrics.to_csv(METRICS_PATH)

    return {"message": "Metrics saved to PVC", "metrics": metrics_data}, 200


@app.route("/evaluate", methods=["GET"])
def run_evaluation():
    """Endpoint to trigger model evaluation."""
    result, status = evaluate_model()
    return jsonify(result), status


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Endpoint to fetch saved metrics."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics file not found. Run evaluation first."}), 404

    df = pd.read_csv(METRICS_PATH, index_col=0)
    metrics_dict = df.to_dict()["Value"]
    
    return jsonify({"metrics": metrics_dict})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)

