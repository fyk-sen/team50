FROM python:3.11

WORKDIR /application

COPY flask/app.py .
COPY flask/templates ./templates
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]