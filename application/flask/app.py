# from flask import Flask, render_template, request, redirect
# import os
# import pandas as pd
# from io import StringIO
# import requests

# app = Flask(__name__, template_folder="templates")

# @app.route('/')
# def index():
#     return render_template('index.html')

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

#         # Save the file as raw.csv in the shared volume
#         input_file_path = "/data/raw_train.csv"
#         df.to_csv(input_file_path, index=False)

#         # Trigger the processing container via HTTP
#         processing_url = "http://processing-service:5001/process"
#         processing_response = requests.post(processing_url)

#         if processing_response.status_code == 200:
#             print("Processing completed successfully.")

#             # **Trigger the model training**
#             training_url = "http://model-service:5002/train"
#             training_response = requests.post(training_url)

#             if training_response.status_code == 200:
#                 print("Model training triggered successfully.")
#             else:
#                 print("Failed to trigger model training.")

#         else:
#             print("Error in processing stage.")

#         # Read the test dataset
#         df_test = pd.read_csv('/data/raw_train.csv')

#         return render_template(
#             'index.html',
#             tables1=[df.to_html(classes='data')],
#             tables2=[df_test.to_html(classes='data')]
#         )

# @app.route("/fetch_results", methods=["GET"])
# def fetch_results():
#     """Fetch predictions, confusion matrix, and classification report from the inference service."""
    
#     # Fetch Metrics from Inference Service
#     inference_url = "http://inference-service:5003/metrics"
#     response = requests.get(inference_url)

#     if response.status_code == 200:
#         results = response.json()

#         classification_report = results.get("classification_report", {})
#         confusion_matrix = results.get("confusion_matrix", {})
        
#         return render_template(
#             "index.html",
#             classification_report=classification_report,
#             confusion_matrix=confusion_matrix
#         )
#     else:
#         return f"Error fetching results: {response.json()}", 500
# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     if 'file' not in request.files:
# #         return redirect(request.url)

# #     file = request.files['file']

# #     if file.filename == '':
# #         return redirect(request.url)

# #     if file:
# #         # Read the file
# #         content = file.read().decode("utf-8")
# #         df = pd.read_csv(StringIO(content))

# #         # Create file directory if not exists
# #         os.makedirs('./data', exist_ok=True)

# #         # Save the file as raw.csv to shared volume
# #         input_file_path = "/data/raw.csv"
# #         df.to_csv(input_file_path, index=False) 

# #         # Trigger the processing container via HTTP
# #         processing_url = "http://processing-service:5001/process"
# #         requests.post(processing_url)


# #         # Temporarily for testing
# #         # df2 = pd.read_csv(StringIO(content))
# #         df_test = pd.read_csv('/data/y_test.csv')

# #         return render_template(
# #             'index.html', 
# #             tables1=[df.to_html(classes='data')], 
# #             titles=df.columns.values,
# #             tables2=[df_test.to_html(classes='data')], 
# #             titles2=df_test.columns.values
# #         )

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True) # Added host and port

import os
import pandas as pd
from flask import Flask, request, render_template, redirect
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = '/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

training_complete = False  # Track model training status

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
    global training_complete
    return render_template('index.html', training_complete=training_complete)

@app.route('/upload', methods=['POST'])
def upload_train_data():
    global training_complete
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
                training_complete = True
                return redirect('/') # Reload page after training
                # return "Processing and Training completed successfully!", 200
            return "Model training failed.", 500
        return "Processing failed.", 500

    return 'Invalid file format. Only CSV files are allowed.', 400

@app.route('/upload_test', methods=['POST'])
def upload_test_data():
    global training_complete

    if not training_complete:
        return "Model training not completed yet.", 400
    
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
            return redirect('/')
            # training_response = training_response_func()
            # if training_response.status_code == 200:
            #     return "Processing and Training completed successfully!", 200
            # return "Model training failed.", 500
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
    app.run(debug=True, port=5000, host='0.0.0.0')