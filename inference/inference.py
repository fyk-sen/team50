# model_infer.py
import joblib
import numpy as np
import os

MODEL_PATH = "/model-storage/trained_model.pkl"

def predict(features):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return None

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([features])
    return prediction[0]

if __name__ == "__main__":
    test_input = [2, 5]
    result = predict(test_input)
    if result is not None:
        print(f"Prediction for input {test_input}: {result}")
