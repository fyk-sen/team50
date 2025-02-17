# model_infer.py
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

MODEL_PATH = "/model-storage/trained_model.pkl"
TEST_DATA_PATH = "/model-storage/test_data.pkl"  # Ensure test data is saved in this path

def predict(features):
    """Predict the output using the trained model."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return None

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([features])
    return prediction[0]

def evaluate_model():
    """Evaluate the model using stored test data."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return
    
    if not os.path.exists(TEST_DATA_PATH):
        print("Test data not found. Provide test data for evaluation.")
        return

    # Load model and test data
    model = joblib.load(MODEL_PATH)
    test_data = joblib.load(TEST_DATA_PATH)
    
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    
    # Get predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred) if hasattr(model, "predict_proba") else None
    mse = mean_squared_error(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")  # Adjust for classification tasks

    print("\nModel Evaluation Metrics:")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    test_input = [2, 5]
    result = predict(test_input)
    if result is not None:
        print(f"Prediction for input {test_input}: {result}")
    
    # Run model evaluation
    evaluate_model()

