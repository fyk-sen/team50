from flask import Flask, render_template, request, redirect
import pandas as pd
from io import StringIO

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
        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))

        # Temporarily for testing
        df2 = pd.read_csv(StringIO(content))

        return render_template(
            'index.html', 
            tables1=[df.to_html(classes='data')], titles=df.columns.values,
            tables2=[df2.to_html(classes='data')], titles2=df2.columns.values
        )

if __name__ == '__main__':
    app.run(debug=True)
