from flask import Flask, render_template, request, redirect
import os
import pandas as pd
from io import StringIO
import requests

app = Flask(__name__, template_folder="templates")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Read the file
        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))

        # Save the file as raw.csv in the shared volume
        input_file_path = "/data/raw_train.csv"
        df.to_csv(input_file_path, index=False)

        # Trigger the processing container via HTTP
        processing_url = "http://processing-service:5001/process"
        processing_response = requests.post(processing_url)

        if processing_response.status_code == 200:
            print("Processing completed successfully.")

            # **Trigger the model training**
            training_url = "http://model-service:5002/train"
            training_response = requests.post(training_url)

            if training_response.status_code == 200:
                print("Model training triggered successfully.")
            else:
                print("Failed to trigger model training.")

        else:
            print("Error in processing stage.")

        # Read the test dataset
        df_test = pd.read_csv('/data/raw_train.csv')

        return render_template(
            'index.html',
            tables1=[df.to_html(classes='data')],
            tables2=[df_test.to_html(classes='data')]
        )

@app.route("/fetch_results", methods=["GET"])
def fetch_results():
    """Fetch predictions, confusion matrix, and classification report from the inference service."""
    
    # Fetch Metrics from Inference Service
    inference_url = "http://inference-service:5003/metrics"
    response = requests.get(inference_url)

    if response.status_code == 200:
        results = response.json()

        classification_report = results.get("classification_report", {})
        confusion_matrix = results.get("confusion_matrix", {})
        
        return render_template(
            "index.html",
            classification_report=classification_report,
            confusion_matrix=confusion_matrix
        )
    else:
        return f"Error fetching results: {response.json()}", 500
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return redirect(request.url)

#     file = request.files['file']

#     if file.filename == '':
#         return redirect(request.url)

#     if file:
#         # Read the file
#         content = file.read().decode("utf-8")
#         df = pd.read_csv(StringIO(content))

#         # Create file directory if not exists
#         os.makedirs('./data', exist_ok=True)

#         # Save the file as raw.csv to shared volume
#         input_file_path = "/data/raw.csv"
#         df.to_csv(input_file_path, index=False) 

#         # Trigger the processing container via HTTP
#         processing_url = "http://processing-service:5001/process"
#         requests.post(processing_url)


#         # Temporarily for testing
#         # df2 = pd.read_csv(StringIO(content))
#         df_test = pd.read_csv('/data/y_test.csv')

#         return render_template(
#             'index.html', 
#             tables1=[df.to_html(classes='data')], 
#             titles=df.columns.values,
#             tables2=[df_test.to_html(classes='data')], 
#             titles2=df_test.columns.values
#         )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Added host and port
