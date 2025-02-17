FROM python:3.11

WORKDIR /processing

COPY processing.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["python", "processing.py"]
