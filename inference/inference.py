from flask import Flask, jsonify
import joblib
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)

MODEL_PATH = "/mnt/model_storage/trained_model.pkl"
X_TEST_PATH = "/data/x_test.csv"
Y_TEST_PATH = "/data/y_test.csv"

def evaluate_model():
    """Runs model evaluation and returns results as JSON."""
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found. Train the model first."}, 404

    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        return {"error": "Processed data files not found. Ensure test data and labels are available."}, 404

    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    Y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred).tolist()
    classification_rep = classification_report(Y_test, Y_pred, output_dict=True)
    predictions = pd.DataFrame({"actual": Y_test, "predicted": Y_pred})

    return {
        "message": "Evaluation completed successfully",
        "classification_report": classification_rep,
        "confusion_matrix": cm,
        "predictions": predictions.to_dict(orient="records")
    }, 200

@app.route("/evaluate", methods=["GET"])
def run_evaluation():
    """Trigger model evaluation and return results."""
    result, status = evaluate_model()
    return jsonify(result), status

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
