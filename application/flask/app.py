from flask import Flask, render_template, request, redirect
import pandas as pd
from io import StringIO
import os
import subprocess

app = Flask(__name__)

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

        # Create file directory if not exists
        os.makedirs('./data', exist_ok=True)

        # Save the file as raw.csv to shared volume
        input_file_path = "./data/raw.csv"
        df.to_csv(input_file_path, index=False) 

        # Trigger processing container
        subprocess.run(['kubectl', 'exec', '-it', 'processing', 'python', 'processing.py'])


        # Temporarily for testing
        # df2 = pd.read_csv(StringIO(content))
        df_test = pd.read_csv('./data/y_test.csv')

        return render_template(
            'index.html', 
            tables1=[df.to_html(classes='data')], titles=df.columns.values,
            tables2=[df_test.to_html(classes='data')], titles2=df_test.columns.values
        )

if __name__ == '__main__':
    app.run(debug=True)
