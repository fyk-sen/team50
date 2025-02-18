FROM python:3.11

WORKDIR /model

COPY model.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5002

CMD ["python", "model.py"]