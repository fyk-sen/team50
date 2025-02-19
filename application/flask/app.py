import os
import pandas as pd
from flask import Flask, request, render_template, redirect
import requests
from io import StringIO

app = Flask(__name__)

UPLOAD_FOLDER = '/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

training_complete = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processing_response_func():
    return requests.post("http://processing-service:5001/process")

def training_response_func():
    return requests.post("http://model-service:5002/train")

@app.route('/')
def index():
    global training_complete
    return render_template('index.html', training_complete=training_complete)

@app.route('/upload', methods=['POST'])
def upload_train_data():
    global training_complete
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        df = pd.read_csv(StringIO(file.read().decode("utf-8")))
        df.to_csv(os.path.join(UPLOAD_FOLDER, "raw_train.csv"), index=False)

        processing = processing_response_func()

        if processing.status_code == 200:
            if training_response_func().status_code == 200:
                training_complete = True
                return redirect('/fetch_results')
            return "Model training failed.", 500
        return "Processing failed.", 500

    return 'Invalid file format. Only CSV files are allowed.', 400

@app.route('/upload_test', methods=['POST'])
def upload_test_data():
    global training_complete

    if not training_complete:
        return "Model training not completed yet.", 400

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        df = pd.read_csv(StringIO(file.read().decode("utf-8")))
        df.to_csv(os.path.join(UPLOAD_FOLDER, "raw_test.csv"), index=False)

        if processing_response_func().status_code == 200:
            return redirect('/fetch_results')

        return "Processing failed.", 500

    return 'Invalid file format. Only CSV files are allowed.', 400

@app.route('/fetch_results', methods=['GET'])
def fetch_results():
    """Fetch evaluation results from inference service."""
    response = requests.get("http://inference-service:5003/evaluate")

    if response.status_code == 200:
        data = response.json()
        return render_template(
            "index.html",
            metrics_table=pd.DataFrame(data["classification_report"]).transpose().to_html(classes='data', index=True),
            cm_table=pd.DataFrame(data["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_html(classes='data', index=True),
            prediction_table=data["predictions"],
            training_complete=True
        )
    
    return f"Error fetching results: {response.json()}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
