from flask import Flask, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Initialize Flask app
app = Flask(__name__)

# File paths
MODEL_PATH = "/mnt/model_storage/trained_model.pkl"
METRICS_PATH = "/data/metrics.csv"
CONFUSION_MATRIX_PATH = "/data/confusion_matrix.csv"
PREDICTIONS_PATH = "/data/predictions.csv"  
X_TEST_PATH = "/data/x_test.csv"
Y_TEST_PATH = "/data/y_test.csv"

def evaluate_model():
    """Runs classification model evaluation, saves full classification metrics, and predictions."""
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found. Train the model first."}, 404

    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        return {"error": "Test data files not found. Ensure test data and labels are available."}, 404

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    Y_test = pd.read_csv(Y_TEST_PATH).values.ravel()  # Flatten if necessary

    # Predict class labels
    Y_pred = model.predict(X_test)

    # Convert predictions: True → 1, False → 0
    if isinstance(Y_pred[0], (bool, np.bool_)):  # Check if predictions are boolean
        Y_pred = Y_pred.astype(int)  # Convert True → 1, False → 0

    # Compute Classification Metrics
    cm = confusion_matrix(Y_test, Y_pred)  # Confusion matrix (NumPy array)
    classification_rep = classification_report(Y_test, Y_pred, output_dict=True)  # Classification report as dict

    # Convert classification report to a structured DataFrame
    df_classification = pd.DataFrame(classification_rep).transpose()
    
    # Convert confusion matrix to DataFrame
    df_confusion = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

    # Save classification report & confusion matrix to CSV
    df_classification.to_csv(METRICS_PATH)
    df_confusion.to_csv(CONFUSION_MATRIX_PATH)

    return {
        "message": "Full classification report, confusion matrix, and predictions saved to PVC",
        "classification_report": classification_rep,
        "confusion_matrix": cm.tolist(),  # Convert to JSON serializable format
        "predictions_path": PREDICTIONS_PATH
    }, 200


@app.route("/evaluate", methods=["GET"])
def run_evaluation():
    """Endpoint to trigger model evaluation."""
    result, status = evaluate_model()
    return jsonify(result), status


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Endpoint to fetch full classification report and confusion matrix."""
    if not os.path.exists(METRICS_PATH) or not os.path.exists(CONFUSION_MATRIX_PATH):
        return jsonify({"error": "Metrics file not found. Run evaluation first."}), 404

    # Load classification report
    df_classification = pd.read_csv(METRICS_PATH, index_col=0)
    classification_dict = df_classification.to_dict()

    # Load confusion matrix
    df_confusion = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
    confusion_dict = df_confusion.to_dict()

    return jsonify({
        "classification_report": classification_dict,
        "confusion_matrix": confusion_dict
    })


@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Endpoint to fetch saved predictions."""
    if not os.path.exists(PREDICTIONS_PATH):
        return jsonify({"error": "Predictions file not found. Run evaluation first."}), 404

    df = pd.read_csv(PREDICTIONS_PATH)
    
    return jsonify({"predictions": df["Predicted"].tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
