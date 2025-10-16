FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PaddleOCR
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create directory for PaddleOCR models
RUN mkdir -p /root/.paddleocr

# Expose port
EXPOSE 8023

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8023/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8023"]
