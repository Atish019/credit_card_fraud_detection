FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the data folder
COPY data /app/data

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
