FROM nvidia/cuda:12.1.0-base-ubuntu22.04
LABEL authors="Claudiu"

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and source code
COPY *.pkl ./
COPY *.pth ./
COPY src/ ./src/
COPY api_service.py .
COPY task_advice_generator.py .
COPY model_loader.py .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the API service
CMD ["python", "api_service.py"]