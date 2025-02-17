FROM python:3.9
WORKDIR /model
COPY model.py .
COPY requirements.txt .
RUN pip install -r requirements.txt .
EXPOSE 5002
CMD ["sh", "-c", "service nginx start && tail -f /dev/null"]
