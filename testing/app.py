import os
import pandas as pd
from flask import Flask, request, render_template, redirect
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the uploaded file is allowed (CSV only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processing_response_func():
    """Trigger data processing via HTTP request."""
    processing_url = "http://processing-service:5001/process"
    return requests.post(processing_url)

def training_response_func():
    """Trigger model training via HTTP request."""
    training_url = "http://model-service:5002/train"
    return requests.post(training_url)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_train_data():
    """Handle training data upload."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        df.to_csv(os.path.join(UPLOAD_FOLDER, "raw_train.csv"), index=False)

        processing_response = processing_response_func()
        if processing_response.status_code == 200:
            training_response = training_response_func()
            if training_response.status_code == 200:
                return "Processing and Training completed successfully!", 200
            return "Model training failed.", 500
        return "Processing failed.", 500

    return 'Invalid file format. Only CSV files are allowed.', 400

@app.route('/upload_test', methods=['POST'])
def upload_test_data():
    """Handle test data upload."""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        df.to_csv(os.path.join(UPLOAD_FOLDER, "raw_test.csv"), index=False)

        processing_response = processing_response_func()
        if processing_response.status_code == 200:
            training_response = training_response_func()
            if training_response.status_code == 200:
                return "Processing and Training completed successfully!", 200
            return "Model training failed.", 500
        return "Processing failed.", 500

    return 'Invalid file format. Only CSV files are allowed.', 400

@app.route('/fetch_results', methods=['GET'])
def fetch_results():
    """Fetch results from the inference service."""
    inference_url = "http://inference-service:5003/metrics"
    response = requests.get(inference_url)

    if response.status_code == 200:
        # Read prediction data
        prediction_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'prediction.csv'))
        metrics_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'metrics.csv'))
        cm_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'confusion_matrix.csv'))

        # Convert data to HTML tables
        prediction_html = prediction_df.to_html(classes='data', index=False)
        metrics_html = metrics_df.to_html(classes='data', index=False)

        # Create confusion matrix image
        cm = cm_df.values
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=cm_df.columns, yticklabels=cm_df.index)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        cm_image_path = os.path.join(UPLOAD_FOLDER, 'confusion_matrix.png')
        plt.savefig(cm_image_path)
        plt.close()

        return render_template(
            "index.html",
            prediction_table=prediction_html,
            metrics_table=metrics_html,
            cm_image=cm_image_path
        )
    return f"Error fetching results: {response.json()}", 500


if __name__ == '__main__':
    app.run(debug=True)