# model_infer.py
import joblib
import numpy as np

MODEL_PATH = "/model/trained_model.pkl"

def predict(features):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([features])
    return prediction[0]

if __name__ == "__main__":
    test_input = [2, 5]
    result = predict(test_input)
    print(f"Prediction for input {test_input}: {result}")
