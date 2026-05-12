# Stage 1: Base Image
# Use an official lightweight Python image. 'slim' is a good balance of size and functionality.
# Using a specific version like 3.11 is recommended for reproducibility.
FROM python:3.11-slim-bookworm

# Set up the working directory in the container
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (curl is needed for the healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and artifacts into the container
COPY ./src ./src
COPY ./artifacts ./artifacts

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application using Gunicorn
# The PYTHONPATH ensures that Python can find modules inside the 'src' directory
CMD ["env", "PYTHONPATH=/app", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "src.api.app:app"]