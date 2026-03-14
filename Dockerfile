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

# Create models directory
RUN mkdir -p /app/final_models

# Copy each model file individually (ensures they're copied)
COPY final_models/construction_defect_detector_best.pth /app/final_models/
COPY final_models/pro_physical_work_ai_mentor_best.pth /app/final_models/
COPY final_models/electrical_defect_detector_best.pth /app/final_models/

# Copy the rest of the application
COPY unified_trade_app.py .

# Verify models were copied and show sizes
RUN echo "=== VERIFYING MODELS ===" && \
    ls -la /app/final_models/ && \
    echo "=== MODEL SIZES ===" && \
    wc -c /app/final_models/*.pth || true

# Expose port
ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 unified_trade_app:app
