# model_infer.py
import joblib
import os
from sklearn.metrics import mean_squared_error, f1_score

MODEL_PATH = "/model-storage/trained_model.pkl"

def predict(features):
    """Predict the output using the trained model."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return None

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([features])
    return prediction[0]

def evaluate_model(model):
    """Evaluate the model using dummy data."""

    # Get predictions
    y_pred = model.predict(X_sample)

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")  # Adjust for classification

    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
    else:
        model = joblib.load(MODEL_PATH)

        # Run a test prediction
        test_input = [2, 5]
        result = predict(test_input)
        if result is not None:
            print(f"Prediction for input {test_input}: {result}")
        
        # Run model evaluation
        evaluate_model(model)
