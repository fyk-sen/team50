FROM python:3.10
COPY model.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5002
CMD ["python", "model.py"]
