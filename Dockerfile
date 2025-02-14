FROM python:3.9
WORKDIR /app
COPY model_train.py .
COPY requirements.txt .
RUN pip install pandas joblib os scikit-learn mlflow
EXPOSE 5002
CMD ["sh", "-c", "service nginx start && tail -f /dev/null"]
