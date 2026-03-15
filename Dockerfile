FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 --retries 5 --upgrade pip && \
    pip install --no-cache-dir --timeout 100 --retries 5 -r requirements.txt

# Copy the application (NO model files - they'll be downloaded from Cloud Storage)
COPY unified_trade_app.py .

# Expose port
ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 unified_trade_app:app
