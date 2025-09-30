# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies required by scientific Python stack.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files into the container image.
COPY . /app

# Install the Meridian package and its runtime dependencies.
RUN pip install --upgrade pip && \
    pip install .

# Expose the port Cloud Run will connect to.
EXPOSE ${PORT}

# Start a very small HTTP server so the container is reachable by Cloud Run.
CMD ["python", "cloud_run_entry.py"]
