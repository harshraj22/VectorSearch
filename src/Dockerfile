FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Don't need to copy app.py since it will be mounted via volume

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]