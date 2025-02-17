FROM python:3.9
WORKDIR /app
COPY model_train.py .
COPY requirements.txt .
RUN pip install requirements.txt .
EXPOSE 5002
CMD ["sh", "-c", "service nginx start && tail -f /dev/null"]
