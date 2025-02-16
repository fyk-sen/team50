FROM python:3.12

WORKDIR /application

COPY /flask_app .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]