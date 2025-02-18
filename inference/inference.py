from flask import Flask, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# File paths
MODEL_PATH = "/data/trained_model.pkl"
METRICS_PATH = "/data/metrics.csv"
PREDICTIONS_PATH = "/data/predictions.csv"  # New file for saving predictions
X_TEST_PATH = "/data/x_test.csv"
Y_TEST_PATH = "/data/y_test.csv"

def evaluate_model():
    """Runs model evaluation, saves metrics, and predictions."""
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

    # Convert predictions: True → 1, False → 0
    if isinstance(Y_pred[0], (bool, np.bool_)):  # Check if predictions are boolean
        Y_pred = Y_pred.astype(int)  # Convert True → 1, False → 0

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

    # Save metrics to PVC (CSV format)
    df_metrics = pd.DataFrame.from_dict(metrics_data, orient="index", columns=["Value"])
    df_metrics.to_csv(METRICS_PATH)

    # Save predictions as CSV
    df_predictions = pd.DataFrame({"Predicted": Y_pred})
    df_predictions.to_csv(PREDICTIONS_PATH, index=False)

    return {
        "message": "Metrics and predictions saved to PVC",
        "metrics": metrics_data,
        "predictions_path": PREDICTIONS_PATH
    }, 200


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


@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Endpoint to fetch saved predictions."""
    if not os.path.exists(PREDICTIONS_PATH):
        return jsonify({"error": "Predictions file not found. Run evaluation first."}), 404

    df = pd.read_csv(PREDICTIONS_PATH)
    
    return jsonify({"predictions": df["Predicted"].tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)

