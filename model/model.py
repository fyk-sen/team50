"""
Model Training Script
"""
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.tree import plot_tree

app = Flask(__name__)

# # Paths
# X_TRAIN = "/data/x_train.csv"
# X_TEST = "/data/x_test.csv"
# Y_TRAIN = "/data/y_train.csv"
# Y_TEST = "/data/y_test.csv"
# MODEL_DIR = "/mnt/model_storage/"
# MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")

# # Ensure model directory exists
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Set up MLflow tracking
# mlflow.set_tracking_uri("http://mlflow-service:5004")
# mlflow.set_experiment("Model_Training")

# @app.route('/train', methods=['POST'])
# def train_model():
#     """
#     Train the ML model
#     """
#     if not os.path.exists(X_TRAIN) or not os.path.exists(Y_TRAIN):
#         return jsonify({"error": "Training data not found"}), 400

#     # Read data
#     x_train = pd.read_csv(X_TRAIN)
#     x_test = pd.read_csv(X_TEST)
#     y_train = pd.read_csv(Y_TRAIN)
#     y_test = pd.read_csv(Y_TEST)

#     # Model parameters
#     n_estimators = 100

#     # Training with MLflow Tracking
#     with mlflow.start_run():
#         print("Training Model...")
#         model = RandomForestRegressor(n_estimators=n_estimators)
#         model.fit(x_train, y_train.values.ravel())

#         # Predictions
#         y_train_pred = model.predict(x_train)
#         y_test_pred = model.predict(x_test)

#         # Compute Metrics
#         train_mse = mean_squared_error(y_train, y_train_pred)
#         train_r2 = r2_score(y_train, y_train_pred)
#         test_mse = mean_squared_error(y_test, y_test_pred)
#         test_r2 = r2_score(y_test, y_test_pred)

#         # Log metrics to MLflow
#         mlflow.log_param("n_estimators", n_estimators)
#         mlflow.log_metric("train_mse", train_mse)
#         mlflow.log_metric("train_r2_score", train_r2)
#         mlflow.log_metric("test_mse", test_mse)
#         mlflow.log_metric("test_r2_score", test_r2)

#         # Log model in MLflow
#         mlflow.sklearn.log_model(model, "trained_rf_model")
#         print("Model logged to MLflow")

#         # Save trained model
#         joblib.dump(model, MODEL_PATH)
#         print(f"Model saved at {MODEL_PATH}")

#     return "Model training completed", 200

# @app.route('/train', methods=['POST'])
# def trigger_training():
#     """Trigger model training when processing is completed."""
#     return train_model()

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5002, debug=True)


app = Flask(__name__)

# Define data paths
X_TRAIN_DATA_PATH = "/data/x_train.csv"
X_TEST_DATA_PATH = "/data/x_test.csv"
Y_TRAIN_DATA_PATH = "/data/y_train.csv"
Y_TEST_DATA_PATH = "/data/y_test.csv"
MODEL_DIR = "/mnt/model_storage/"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
TREE_IMAGE_PATH = os.path.join(MODEL_DIR, "rf_tree.png")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(MODEL_DIR, "roc_curve.png")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Set up MLflow tracking
mlflow.set_tracking_uri("http://mlflow-service:5004")
mlflow.set_experiment("Model_Training")

def train_model():
    """Train a model and log it to MLflow."""
    if not os.path.exists(X_TRAIN_DATA_PATH):
        print("No Data available")
        return "No Data available", 400

    # Read data
    x_train = pd.read_csv(X_TRAIN_DATA_PATH)
    x_test = pd.read_csv(X_TEST_DATA_PATH)
    y_train = pd.read_csv(Y_TRAIN_DATA_PATH).values.ravel()
    y_test = pd.read_csv(Y_TEST_DATA_PATH)

    # Training parameters
    n_estimators = 100

    # Start MLflow run
    with mlflow.start_run():
        print("Training Model...")
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(x_train, y_train)

        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Compute Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        error_rate = 1 - test_accuracy
        f1 = f1_score(y_test, y_test_pred)
        
        # **Confusion Matrix**
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close()

        # **ROC Curve**
        y_test_proba = model.predict_proba(x_test)[:, 1]  # Probability of positive class
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(ROC_CURVE_PATH)
        plt.close()

        # **Decision Tree Visualization (first tree from RandomForest)**
        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0], feature_names=x_train.columns, filled=True, fontsize=8)
        plt.savefig(TREE_IMAGE_PATH, dpi=300, bbox_inches="tight")
        plt.close()

        # **Log metrics in MLflow**
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("error_rate", error_rate)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # **Log artifacts (confusion matrix, ROC curve, Decision Tree)**
        mlflow.log_artifact(CONFUSION_MATRIX_PATH)
        mlflow.log_artifact(ROC_CURVE_PATH)
        mlflow.log_artifact(TREE_IMAGE_PATH)
        print("Decision Tree, Confusion Matrix, and ROC Curve saved and logged to MLflow.")

        # Verify image exists before logging
        if os.path.exists(TREE_IMAGE_PATH):
            mlflow.log_artifact(TREE_IMAGE_PATH)
            mlflow.log_artifact(ROC_CURVE_PATH)
            mlflow.log_artifact(TREE_IMAGE_PATH)
            print("Decision Tree, Confusion Matrix, and ROC Curve saved and logged to MLflow.")
        else:
            print(f"Error: Decision Tree image not found at {TREE_IMAGE_PATH}")

        # Log the model in MLflow
        mlflow.sklearn.log_model(model, "trained_rf_model")
        print("Model logged to MLflow")

        # Save model to shared PVC
        joblib.dump(model, MODEL_PATH)
        print(f"Model has been saved at {MODEL_PATH}")

    return "Model training completed", 200

@app.route('/train', methods=['POST'])
def trigger_training():
    """Trigger model training when processing is completed."""
    return train_model()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)